# PRISM RINGS 2.0: implemented mathematics and novel system concepts

This document distinguishes working prototype mechanisms from ideas that would need model training or benchmarking.

## 1. SpectraForge color fingerprint (implemented)

For each memory \(m\) and channel \(k\), a deterministic, editable rule-based lexical score \(s_k(m)\) is calculated. The channels are code, writing, tasks, computer, research, procedural, dialogue, creative, system, causal.

\[
c_k(m) = \frac{\exp((s_k(m)-s_{\max})/\tau)}{\sum_j\exp((s_j(m)-s_{\max})/\tau)},\qquad \sum_k c_k=1
\]

The color temperature \(\tau\) is adjustable in the Ring Observatory. A perceptual RGB ring label is composed by weighted channel blending and ring-depth darkening:

\[
RGB_r(m) = (1-0.045r)\sum_k c_k(m)\,RGB_k
\]

This is a visualization/routing heuristic, **not a learned or validated hyperdimensional encoder**.

## 2. Local spectral/lexical vectors (implemented)

Features include word tokens and character trigrams; stable BLAKE2b hashes map each feature into a signed 246-dimensional lexical lane. Ten channel weights occupy ten additional dimensions.

\[
v(m)=\operatorname{normalize}\left([\sum_{f:h(f)=i}\operatorname{sign}(f)]_{i=1}^{246}\oplus 10\mathbf c(m)\right)
\]

The resulting 256-dimensional vector can be queried in embedded Weaviate. These are reproducible lexical/spectral vectors and may be less semantically accurate than pretrained embeddings.

## 3. Ring Resonance (partial prototype)

For query \(q\) and stored memory \(m\), SQLite performs local vector similarity and adjusts for verification, importance, project scope, and Weaviate candidates:

\[
R(q,m)=\cos(v(q),v(m))+0.3C_{W}(m)+0.05I_m+0.05V_m+0.10P_m
\]

The Weaviate rank bonus \(C_W\) is **not** a replacement for source-authority validation: every candidate is checked for matching UUID, revision, consent and deletion state in SQLite.

## 4. HoloRing evidence capsules (implemented, extractive)

Let \(M\) be a source memory set. Capsule identity is determined by source UUIDs and source revisions:

\[
ID_{capsule}=SHA256\left(\operatorname{sort}([(id_m,revision_m)]_{m\in M})\right)
\]

The capsule includes actual excerpt text and `UUID@revision` anchors, not an invented LLM paraphrase. A source-linked capsule is *stale* if any source is tombstoned or its latest revision differs from the anchored revision.

## 5. ChronoOrbit priority (implemented)

\[
P_i(t)=0.6p_i+2\sigma\left(-\frac{d_i-t}{12\,hours}\right)+0.2\min(wait\_days_i,30)+0.4unblocks_i
\]

Tasks are filtered by due time and completion of all parents **before** priority ranking; agent work requires explicit approval. Recurrence advances only after successful execution.

## 6. Crash-safe outbox and encrypted index reconciliation (implemented)

Every index operation writes a durable SQLite outbox record transactionally with the memory mutation. UUID + monotone revision identify each source record:

\[
\operatorname{Accept}(w,m) \Longleftrightarrow
 UUID_w=UUID_m\ \land\ Revision_w=Revision_m\ \land\neg Deleted_m\ \land\ Consent_m
\]

Weaviate is a rebuildable cache; SQLite owns the truth. On restart, all records are eligible for reconciliation to avoid losing recent updates that never reached the last encrypted Weaviate snapshot.

## 7. Vault-sealed embedded index (implemented)

\[
K_w=HKDF_{SHA256}(K_{GUI},\text{info}=\texttt{humoid-prism-embedded-weaviate-v1})
\]

\[
Archive=\texttt{MAGIC}\Vert Nonce\Vert AES256GCM_{K_w}(Nonce,\text{tar.gz(index)}, AAD=\texttt{HAG1})
\]

All on-disk snapshots use fresh random nonces, integrity tags, and atomic writes. While running, the embedded Weaviate datastore is only allowed in private Linux tmpfs and the local endpoints require a fresh ephemeral API key. Re-encryption of the index is included in GUI master-key rotation. This does **not** encrypt live RAM, swap, in-flight API responses, or a remote Docker volume.

## Experimental future ideas (not claimed as implemented)

- **Harmonic vector binding:** independent learned code/write/task modalities with randomized complex phase keys and spectral unbinding.
- **Counterfactual skill rehearsal:** simulated workflow trajectories contrasted against failures and user approvals.
- **Cross-domain memory diffraction:** subquery-specific neural encoders and graph-based fusion.
- **Automated summary fidelity audits:** model-scored entailment and constraints preservation against evidence.
- **Full OS background schedules:** operating-system service integration, separate from the current GUI-only due alerts.
