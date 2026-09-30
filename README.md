Nume: Costescu Nicolas-Cristian
Grupă: 333CD

# Tema 2 ASC - CUDA Merkle Tree & Proof of Work

## Organizare

**Explicație pentru soluția aleasă:**
Soluția mea a urmărit portarea calculului SHA256 pe GPU, minimizând pe cât posibil instrucțiunile costisitoare din interiorul buclelor critice (hot loops) și utilizând ierarhia de memorie CUDA pentru a reduce latențele.

Pentru **Căutarea Nonce-ului (Proof of Work)** am aplicat următoarele optimizări majore:
*   **Midstate Optimization (Salvarea contextului SHA256):** Aceasta a adus cel mai mare spor de performanță. Deoarece începutul blocului (content-ul) este mereu același, am apelat `sha256_init` și `sha256_update` pe block content *o singură dată*, înainte de bucla `for`. În interiorul buclei copiez doar acest context parțial salvat (`base_ctx`) și adaug doar lungimea nonce-ului curent. Astfel, scutesc GPU-ul de hașurarea repetată a datelor statice la fiecare iterație.
*   **Oprirea imediată a thread-urilor (Volatile):** Pentru a preveni calculul inutil după găsirea soluției, am folosit `volatile int* v_found = found;`. Fără `volatile`, compilatorul `nvcc` memora valoarea lui `found` în regiștri, făcând restul thread-urilor să ruleze în gol până la `max_nonce` chiar dacă soluția fusese găsită. Prin `volatile` forțez citirea din memoria globală/L2 cache, ceea ce oprește toate thread-urile la momentul oportun (atingând timpi de ~0.07s).
*   **Utilizarea Memoriei Shared și Local:** Conținutul blocului a fost adus în `__shared__ memory` (`shared_content`) pentru ca toate thread-urile dintr-un block să îl acceseze rapid. Dificultatea (string-ul de 64 de caractere) a fost salvată în `local memory / regiștri` (`local_diff`).
*   **Funcția `intToString` custom și comparație "On-the-fly":** Funcțiile precum `sprintf` și `strcmp` sunt lente pe GPU. Am implementat o funcție secvențială ușoară de conversie din număr în string. La final, hash-ul nu este stocat complet pentru a fi comparat; în schimb, calculez caracterele hexazecimale pe loc cu operații pe biți (`>> 4` și `& 0x0F`) și le compar direct cu `local_diff`, dând `break` la prima diferență (Early Exit).
*   **CUDA Streams:** Lansarea kernelului pentru nonce se face folosind `cudaStream_t` și `cudaMemcpyAsync`, ascunzând o parte din latența de transfer.

Pentru calculul **Arborelui Merkle (Merkle Root)**:
*   Am folosit o abordare iterativă cu două kernel-uri. Primul (`hash_transactions_kernel`) calculează hash-urile frunzelor de bază complet în paralel.
*   Al doilea (`build_merkle_level_kernel`) este lansat într-o buclă pe CPU (`while (n > 1)`), înjumătățind `n`-ul la fiecare pas. Tratarea numărului impar de noduri se face direct pe device, verificând dacă `i * 2 + 1 < n` și duplicând ultimul nod în caz contrar.

**Consideri că tema este utilă?**
Foarte utilă. Este un exercițiu excelent pentru a înțelege cum detaliile arhitecturale (memory hierarchy, register allocation, branch divergence) pot influența masiv timpul de execuție al unui algoritm aparent banal.

**Consideri implementarea naivă, eficientă, se putea mai bine?**
Implementarea curentă este foarte eficientă, profitând de *Midstate Optimization* și *Early Exit*. Se putea optimiza și mai mult convertind dificultatea direct într-un array de 32 de byți pe CPU pentru a elimina total conversia hexazecimală din GPU, și aplicând `__launch_bounds__` pentru controlul strict al regiștrilor, dar performanța actuală se încadrează deja foarte bine în timpii limită.

## Dificultăți întâmpinate
*   **Oprirea blocurilor divergente:** Cea mai mare problemă a fost ca restul GPU-ului să afle că un thread a găsit soluția. Modificarea flag-ului folosind `atomicExch` mergea pe memorie, dar thread-urile citeau dintr-o valoare pusă în regiștri de optimizatorul compilatorului. Introducerea pointer-ului `volatile` a fost necesară pentru a garanta sincronizarea "vizuală" a memoriei.
*   **Presiunea pe memorie (Contextul SHA256):** Inițializarea și stocarea bufferelor de SHA256 este intensivă; trecerea pe `shared_memory` a conținutului a ajutat la reducerea traficului pe memoria globală.

## Lucruri interesante descoperite pe parcurs
*   Costul uriaș al recalculării hașurilor pe prefixuri statice. Salvarea stării SHA256 (`base_ctx`) înainte de intrarea în buclă taie efectiv zeci de mii de operații pe fiecare thread.
*   Importanța detaliilor legate de compilatorul `nvcc` și modul în care alege să facă *caching* agresiv al variabilelor.

## Resurse utilizate
*   Laboratoarele de ASC (CUDA - Shared Memory, Streams).
*   Documentația NVIDIA (C++ Programming Guide) cu privire la atributele variabilelor (`volatile`) și operațiile atomice.
*   **LLM Utilizat:** Gemini 3.1 Pro - folosit pentru asistență în identificarea bottleneck-urilor hardware.

### Interacțiunea cu uneltele de tip LLM (Gemini)
*Uneltele LLM au fost folosite strict pentru a dezbate optimizări de micro-arhitectură GPU, după ce logica de bază a fost scrisă.*

**Prompt (Eu):**
> Am implementat algoritmul PoW în CUDA dar am un timp de 0.15s. Cum pot scădea acest timp? Kernelul are o buclă for cu un `if (*found == 1) return;` dar pare să nu se oprească la timp.
**Răspuns Gemini (Sintetizat):**
> Problema principală este că variabila `found` este stocată în regiștri de către compilator. Când un thread modifică memoria globală, celelalte nu văd schimbarea imediat. Trebuie să declari citirea ca `volatile int* v_found = found;` pentru a forța compilatorul să citească din L2 Cache/Global Memory la fiecare iterație. A doua optimizare critică este să scoți `sha256_init` și partea de `update` a conținutului blocului în afara buclei `for` (Midstate Optimization).

**De ce a fost util:** Explicația despre `volatile` a fost esențială. Am rescris manual bucla adăugând `base_ctx` și pointerul `volatile`, iar timpul de căutare a nonce-ului a scăzut dramatic, permițând terminarea imediată a kernelului la găsirea primei soluții valide.

