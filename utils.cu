#include <stdio.h>
#include <stdint.h>
#include "utils.h"
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ int intToString(uint64_t num, char* out) {
    if (num == 0) {
        out[0] = '0';
        out[1] = '\0';
        return 1;
    }

    int i = 0;
    while (num != 0) {
        int digit = num % 10;
        num /= 10;
        out[i++] = '0' + digit;
    }

    for (int j = 0; j < i / 2; j++) {
        char temp = out[j];
        out[j] = out[i - j - 1];
        out[i - j - 1] = temp;
    }
    out[i] = '\0';
    return i;
}

__host__ __device__ void apply_sha256_opt(const BYTE *input, size_t input_length, BYTE *output) {
    SHA256_CTX ctx;
    BYTE buf[SHA256_BLOCK_SIZE];
    const char hex_chars[] = "0123456789abcdef";

    sha256_init(&ctx);
    sha256_update(&ctx, input, input_length);
    sha256_final(&ctx, buf);

    for (size_t i = 0; i < SHA256_BLOCK_SIZE; i++) {
        output[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
        output[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
    }
    output[SHA256_BLOCK_SIZE * 2] = '\0';
}

__host__ __device__ int compare_hashes(BYTE* hash1, BYTE* hash2) {
    for (int i = 0; i < SHA256_HASH_SIZE - 1; i++) {
        if (hash1[i] < hash2[i]) {
            return -1; 
        } else if (hash1[i] > hash2[i]) {
            return 1; 
        }
    }
    return 0; 
}

__global__ void hash_transactions_kernel(int transaction_size, BYTE *transactions, int n, BYTE *hashes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        apply_sha256_opt(transactions + i * transaction_size, transaction_size - 1, hashes + i * SHA256_HASH_SIZE);
    }
}

__global__ void build_merkle_level_kernel(int n, BYTE *hashes_in, BYTE *hashes_out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int new_n = (n + 1) / 2;
    
    if (i < new_n) {
        BYTE combined[128]; 
        
        for (int j = 0; j < 64; j++) {
            combined[j] = hashes_in[i * 2 * SHA256_HASH_SIZE + j];
        }

        if (i * 2 + 1 < n) {
            for (int j = 0; j < 64; j++) {
                combined[64 + j] = hashes_in[(i * 2 + 1) * SHA256_HASH_SIZE + j];
            }
        } else {
            for (int j = 0; j < 64; j++) {
                combined[64 + j] = hashes_in[i * 2 * SHA256_HASH_SIZE + j];
            }
        }

        apply_sha256_opt(combined, 128, hashes_out + i * SHA256_HASH_SIZE);
    }
}

__global__ void find_nonce_kernel(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash_out, uint32_t *valid_nonce, int *found) {
    __shared__ char shared_content[256];
    

    if (threadIdx.x < current_length) {
        shared_content[threadIdx.x] = block_content[threadIdx.x];
    }
    
    __syncthreads(); 

    char local_diff[64];

    for (int i = 0; i < 64; i++) {
        local_diff[i] = difficulty[i];
    }

    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    const char hex_chars[] = "0123456789abcdef";

    SHA256_CTX base_ctx;
    sha256_init(&base_ctx);
    sha256_update(&base_ctx, (BYTE*)shared_content, current_length);    

    volatile int* v_found = found;
    int iter = 0;
    for (uint32_t nonce = index; nonce < max_nonce; nonce += stride) {
        
        if (*v_found == 1) return; 

        char nonce_string[NONCE_SIZE];
        int nonce_len = intToString(nonce, nonce_string);

        SHA256_CTX ctx = base_ctx;
        sha256_update(&ctx, (BYTE*)nonce_string, nonce_len);
        
        BYTE buf[SHA256_BLOCK_SIZE]; 
        sha256_final(&ctx, buf);

        bool is_valid = true;
        
        for (int i = 0; i < 32; i++) {
            char c1 = hex_chars[(buf[i] >> 4) & 0x0F];
            if (c1 > local_diff[i * 2]) { is_valid = false; break; }
            if (c1 < local_diff[i * 2]) { break; } 
            
            char c2 = hex_chars[buf[i] & 0x0F];
            if (c2 > local_diff[i * 2 + 1]) { is_valid = false; break; }
            if (c2 < local_diff[i * 2 + 1]) { break; }
        }

        if (is_valid) {
            if (atomicExch(found, 1) == 0) {
                *valid_nonce = nonce;
                for(int i = 0; i < SHA256_BLOCK_SIZE; i++) {
                    block_hash_out[i * 2]     = hex_chars[(buf[i] >> 4) & 0x0F];
                    block_hash_out[i * 2 + 1] = hex_chars[buf[i] & 0x0F];
                }
                block_hash_out[SHA256_BLOCK_SIZE * 2] = '\0';
            }
            return;
        }
    }
}

void construct_merkle_root(int transaction_size, BYTE *transactions, int max_transactions_in_a_block, int n, BYTE merkle_root[SHA256_HASH_SIZE]) {
    BYTE *d_transactions, *d_hashes_in, *d_hashes_out;
    
    cudaMalloc((void**)&d_transactions, n * transaction_size);
    cudaMalloc((void**)&d_hashes_in, max_transactions_in_a_block * SHA256_HASH_SIZE);
    cudaMalloc((void**)&d_hashes_out, max_transactions_in_a_block * SHA256_HASH_SIZE);

    cudaMemcpy(d_transactions, transactions, n * transaction_size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hash_transactions_kernel<<<blocks, threads>>>(transaction_size, d_transactions, n, d_hashes_in);

    while (n > 1) {
        int new_n = (n + 1) / 2;
        blocks = (new_n + threads - 1) / threads;
        
        build_merkle_level_kernel<<<blocks, threads>>>(n, d_hashes_in, d_hashes_out);

        BYTE *temp = d_hashes_in;
        d_hashes_in = d_hashes_out;
        d_hashes_out = temp;

        n = new_n;
    }

    cudaMemcpy(merkle_root, d_hashes_in, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);

    cudaFree(d_transactions);
    cudaFree(d_hashes_in);
    cudaFree(d_hashes_out);
}

int find_nonce(BYTE *difficulty, uint32_t max_nonce, BYTE *block_content, size_t current_length, BYTE *block_hash, uint32_t *valid_nonce) {
    BYTE *d_difficulty, *d_block_content, *d_block_hash;
    uint32_t *d_valid_nonce;
    int *d_found;
    int h_found = 0;
    cudaStream_t stream1;
    
    cudaStreamCreate(&stream1);
    cudaMalloc((void**)&d_difficulty, SHA256_HASH_SIZE);
    cudaMalloc((void**)&d_block_content, current_length + NONCE_SIZE); 
    cudaMalloc((void**)&d_block_hash, SHA256_HASH_SIZE);
    cudaMalloc((void**)&d_valid_nonce, sizeof(uint32_t));
    cudaMalloc((void**)&d_found, sizeof(int));

    cudaMemcpyAsync(d_difficulty, difficulty, SHA256_HASH_SIZE, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_block_content, block_content, current_length + 1, cudaMemcpyHostToDevice, stream1); 
    cudaMemcpyAsync(d_found, &h_found, sizeof(int), cudaMemcpyHostToDevice, stream1);

    int threads = 256;
    int blocks = 512; 

    find_nonce_kernel<<<blocks, threads, 0, stream1>>>(d_difficulty, max_nonce, d_block_content, current_length, d_block_hash, d_valid_nonce, d_found);
    cudaStreamSynchronize(stream1);
    cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost);
    
    if (h_found == 1) {
        cudaMemcpy(valid_nonce, d_valid_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(block_hash, d_block_hash, SHA256_HASH_SIZE, cudaMemcpyDeviceToHost);
    }

    cudaStreamDestroy(stream1);
    cudaFree(d_difficulty);
    cudaFree(d_block_content);
    cudaFree(d_block_hash);
    cudaFree(d_valid_nonce);
    cudaFree(d_found);

    return h_found ? 0 : 1;
}

__global__ void dummy_kernel() {}

void warm_up_gpu() {
    BYTE *dummy_data;
    cudaMalloc((void **)&dummy_data, 256);
    dummy_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaFree(dummy_data);
}
