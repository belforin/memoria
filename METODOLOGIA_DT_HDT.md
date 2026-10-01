# Metodología: DT vs HDT (flujo único vs flujos separados) sobre D4RL

## 0. Contexto: qué se traslada del trabajo de referencia y qué no

El trabajo de tu compañero compara dos topologías Transformer (unistream vs
jerárquica) bajo **MaskDP**: un modelo de autoencoding enmascarado
bidireccional, entrenado con múltiples ratios de máscara, del que se extrae
una política *zero-shot* interpretando el patrón de máscara como interfaz de
tarea (Algoritmo 1 de su metodología). Nuestro caso es estructuralmente
distinto en un punto clave:

- **MaskDP**: objetivo de reconstrucción bidireccional (sin causalidad),
  la política se obtiene sin reentrenar, reinterpretando el enmascaramiento
  en cada paso de control.
- **DT/HDT** (`agent/dt.py`, `agent/hdt.py`): objetivo autoregresivo causal,
  condicionado a un *return-to-go*, entrenado directamente como clonación de
  comportamiento acción-a-acción (Ecuación de pérdida MSE en
  `DTAgent.update_actor`). No hay esquema de máscaras ni Algoritmo 1 propio
  — la política ya es nativa del modelo, sin paso de conversión.

Por eso la metodología no se copia literal: se traslada la **estructura en
etapas** (validación propioceptiva → régimen visual → interpretabilidad) y
la **disciplina experimental** (paridad de parámetros, pre-registro de
protocolo, agregación por semillas), pero el contenido de cada etapa se
adapta al hecho de que DT/HDT son causales y ya producen una política
directamente.

Lo que ya tenemos construido, en términos de su Figura 3.1:

| Su trabajo | Nuestro repo | Equivalencia |
|---|---|---|
| Unistream (MaskDP) | `agent/dt.py` (`DecisionTransformer`) | línea base de flujo único |
| Jerárquica, fusión por auto-atención conjunta | `agent/hdt.py` (`HierarchicalDecisionTransformer`) | `ObservationSequenceEncoding` + `ActionSequenceEncoding` (cada uno su propio stack causal) → tokens `r,s,a` intercalados → stack superior compartido (`self.blocks`) |
| Fusión por atención cruzada simétrica | *no existe todavía* | pendiente si se quiere esa variante |
| Paridad de parámetros entre variantes | *no existe todavía* | `dt.yaml`/`hdt.yaml` hoy usan `n_embd=128` en ambos → HDT sale con ~2x los parámetros de DT (medido en el smoke test: DT 1.123.846 vs HDT 2.172.934) |

### 0.1 Aclaración de objetivo (confirmada con el usuario, 2026-09-07)

Se planteó la duda de si el objetivo es comparar HDT contra DT (ambos
propios) o comparar HDT directamente contra el modelo real de Benjamín.
Investigando el repo se encontró que el trabajo real de Benjamín SÍ está
versionado acá, en ramas separadas (no solo descrito en prosa en este
documento):

- `origin/MaskDP_paper` (tag `results-uni`): MaskDP **unistream**,
  propioceptivo sobre DMC.
- `origin/hierarchical` (tag `results-proprio-hier`): MaskDP
  **jerárquico**, propioceptivo sobre DMC. Acá `agent/mdp.py`
  (`MaskedDPMultimodal`) sí implementa fusión por **atención cruzada
  real** (`CoAttentionBlock`/`ParallelCoAttentionBlock`), a diferencia de
  nuestro `hdt.py` (solo auto-atención conjunta) — confirma lo anotado en
  la Tabla de la sección 0 original.
- También existen ramas de Procgen (`hier-procgen*`) e interpretabilidad
  (`*-attattr`, `*-sarfa`) para ambas topologías.
- **No hay checkpoints ni números de resultados versionados en git**
  (gitignoreados, como es esperable) — solo el código, no sus pesos
  entrenados ni tablas de resultados.

Decisión confirmada con el usuario: por ahora se sigue trabajando en
**DT vs HDT (propios)** como ya está armado en este documento — DT es el
análogo a su unistream, HDT es el análogo a su jerárquica (ver tabla en
la sección 0). La comparación directa contra los números reales de
Benjamín se hace **después**, cuando el usuario aporte su informe/tesis
con los resultados (retornos, tablas) — todavía no localizado en el
filesystem de este cluster. No se va a re-entrenar el código real de
Benjamín (`agent/mdp.py` en `hierarchical`/`MaskDP_paper`) a menos que se
decida lo contrario más adelante.

## 1. Etapa 0 — Paridad de parámetros (control experimental previo)

Antes de comparar rendimiento, hay que igualar presupuesto de parámetros,
igual que su Tabla 3.1. Pasos:

1. Medir el conteo de parámetros de DT en su config actual (`agent/dt.yaml`,
   `n_embd=128`, `n_layer=5`) — ya sabemos que da 1.123.846 (impreso por
   `DTAgent.__init__`, línea `print("number of parameters: %e", ...)`).
2. Para HDT, reducir `n_embd` de los sub-encoders (`n_obs_layer`,
   `n_act_layer`) y/o del bloque superior (`n_layer`) hasta acercarse a ese
   número, análogo a sus variantes `self_attn` (única reducción uniforme,
   ya que nuestra fusión actual es por auto-atención conjunta, no cruzada).
3. Documentar la config resultante en una tabla como su Tabla 3.1
   (`dim. codificadores`, `dim. fusión`, `parámetros`, `% de desviación
   respecto a DT`).
4. Opcional, si se quiere replicar también su comparación de mecanismos de
   fusión: implementar una variante `hdt` con atención cruzada simétrica
   (`CrossAttention`/`CoAttentionBlock` en su nomenclatura) en vez de la
   auto-atención conjunta actual, y repetir la paridad para esa variante.
   Esto es trabajo nuevo, no existe en `agent/modules/attention.py` hoy
   (solo está `Block`, que es auto-atención causal simple).

### 1.1 Resultado (config aplicada en `agent/hdt.yaml`)

Búsqueda por barrido de `n_embd` (manteniendo `n_obs_layer=n_act_layer=
n_layer=3`, la profundidad ya definida en `hdt.yaml`, es decir la
"reducción uniforme" mencionada arriba) contra el conteo real de
parámetros de `DecisionTransformer`/`HierarchicalDecisionTransformer`
(`agent/dt.py`, `agent/hdt.py`), sobre las tres tareas de la Etapa 1:

| Tarea | obs/action | DT (`n_embd=128,n_layer=5`) | HDT (`n_embd=88,n_obs=n_act=n_top=3`) | Desviación |
|---|---|---|---|---|
| halfcheetah | 17 / 6 | 1.123.846 | 1.113.734 | -0.90% |
| walker2d | 17 / 6 | 1.123.846 | 1.113.734 | -0.90% |
| hopper | 11 / 3 | 1.122.307 | 1.112.675 | -0.86% |

`n_embd=88` es el valor que minimiza la desviación dentro de la grilla de
valores pares probados (20 a 128), y la desviación es prácticamente
constante entre tareas (~-0.9%), así que sirve como config única para las
tres. Ya aplicado en `agent/hdt.yaml`. No hizo falta tocar las
profundidades (`n_obs_layer`/`n_act_layer`/`n_layer` se mantienen en 3)
ni la opción de atención cruzada (punto 4, sigue pendiente/opcional).

## 2. Etapa 1 — Validación en control propioceptivo (D4RL)

Reemplaza su DMC (cheetah/walker/quadruped, datos *near-expert* propios de
MaskDP) por **D4RL mujoco-gym**, ya cacheado en `~/.d4rl/datasets/`:
`halfcheetah`, `hopper`, `walker2d`, cada uno en `-expert-v2` (y también
tenemos `-medium-v2`/`-medium_replay-v2` por si se decide ampliar el rango
de calidad de datos, algo que ellos deliberadamente no hicieron porque
usan BC puro — nosotros también usamos BC puro en DT/HDT, así que aplica
el mismo argumento: **entrenar solo contra `-expert-v2`** para no enseñarle
al modelo a clonar comportamiento subóptimo).

### 2.1 Entorno y datos
- Igual que ellos seleccionaron 3 de 7 tareas DMC por rango de dificultad,
  nosotros podemos usar las 3 tareas D4RL mujoco-gym que ya están
  descargadas (halfcheetah, hopper, walker2d) para cubrir un rango de
  dimensionalidad de observación/acción similar en espíritu a su Tabla 3.6:

  | Tarea | obs | action |
  |---|---|---|
  | halfcheetah | 17 | 6 |
  | hopper | 11 | 3 |
  | walker2d | 17 | 6 |

  (halfcheetah ya convertido con `d4rl_data.py`; hopper/walker2d requieren
  el mismo paso de conversión, ya soportado por el script sin cambios —
  solo hace falta correrlo con `--domain hopper`/`--domain walker2d`).

  **Actualización — ya convertidos los tres:**
  ```
  python d4rl_data.py --hdf5 ~/.d4rl/datasets/hopper_expert-v2.hdf5 --out data/hopper_expert --domain hopper
  python d4rl_data.py --hdf5 ~/.d4rl/datasets/walker2d_expert-v2.hdf5 --out data/walker2d_expert --domain walker2d
  ```

  | Dataset | Episodios guardados | obs_dim | action_dim | Ruta |
  |---|---|---|---|---|
  | halfcheetah-expert-v2 | 1000 | 17 | 6 | `data/halfcheetah_expert/halfcheetah/` |
  | hopper-expert-v2 | 1027 | 11 | 3 | `data/hopper_expert/hopper/` |
  | walker2d-expert-v2 | 1000 | 17 | 6 | `data/walker2d_expert/walker2d/` |

  (hopper tiene más episodios que halfcheetah/walker2d porque sus
  episodios terminan antes por `terminal`/`timeout`, en vez de agotar
  siempre el horizonte de 1000 pasos — consistente con que hopper es
  la tarea más inestable de las tres).

### 2.1.1 Qué recibe el modelo como observación (sin imágenes)
En Etapa 1 el modelo **no ve imágenes**: recibe el vector de estado
propioceptivo crudo que trae cada `.hdf5` de D4RL (`d4rl_data.py` solo
copia el array `observations`, sin procesarlo). Los módulos
`agent/modules/pixel_encoder.py` e `impala_cnn.py` existen en el repo
pero no se usan en este pipeline — están reservados para la Etapa 2
visual (CoinRun/Procgen, ver §3).

Siguiendo la convención estándar de Gym-MuJoCo (posición x del torso
excluida del vector de observación), cada dimensión corresponde a:

| Dataset | obs_dim | Contenido | action_dim | Contenido |
|---|---|---|---|---|
| halfcheetah | 17 | qpos[1:] (8: altura z + ángulo torso + 6 ángulos articulares: bthigh/bshin/bfoot/fthigh/fshin/ffoot) + qvel (9: vel. x/z del torso + vel. angular torso + 6 vel. angulares articulares) | 6 | torques continuos por articulación (bthigh, bshin, bfoot, fthigh, fshin, ffoot) |
| hopper | 11 | qpos[1:] (5: altura z + ángulo torso + 3 ángulos articulares: thigh/leg/foot) + qvel (6: vel. x/z torso + vel. angular torso + 3 vel. angulares articulares) | 3 | torques continuos (thigh, leg, foot) |
| walker2d | 17 | qpos[1:] (8: altura z + ángulo torso + 6 ángulos articulares: thigh/leg/foot × pierna izq/der) + qvel (9: vel. análogas) | 6 | torques continuos (thigh, leg, foot × pierna izq/der) |

En resumen: es "espacio articulado" (ángulos + velocidades angulares del
cuerpo simulado, más altura/velocidad del torso), no video ni imágenes.

### 2.2 Objetivo de entrenamiento
Ya implementado: MSE entre acción predicha y acción real
(`DTAgent.update_actor`), condicionado por return-to-go escalado
(`return_scale`). A diferencia de su Ecuación 3.3 (dos términos, estado +
acción, porque MaskDP reconstruye ambas modalidades), DT/HDT **solo
predicen acción** — no hay término de reconstrucción de estado. Esto es
una diferencia estructural del método, no algo que haya que igualar.

### 2.3 Sin paso de "evaluación de retorno sin reentrenamiento"
Su Sección 3.2.3 (Algoritmo 1: enmascaramiento iterativo) no aplica —
DT/HDT ya ejecutan en lazo cerrado de forma nativa vía `DTAgent.act()`
(implementado y probado en `eval_dt.py`: se mantiene un buffer deslizante
de `obs/action/rtg/timestep`, se recondiciona el return-to-go restando la
recompensa recibida en cada paso). Este mecanismo ya está construido y
verificado con el smoke checkpoint.

### 2.4 Protocolo de evaluación
Adaptado de su Sección 3.2.4, quitando la tarea de *goal reaching* (D4RL
mujoco-gym no tiene metas, solo maximización de retorno):

- **Única tarea**: maximización de retorno en lazo cerrado (closed-loop),
  vía `eval_dt.py`, con horizonte de ejecución = 1 (replanifica cada paso,
  igual que ellos).
- **Métrica**: en vez de retorno crudo, usar el **score normalizado D4RL**
  (`100 * (retorno - retorno_random) / (retorno_experto - retorno_random)`),
  que es el estándar de la literatura de D4RL/Decision Transformer y
  permite comparar entre tareas de escalas de recompensa muy distintas
  (halfcheetah, hopper, walker2d no son comparables en retorno crudo). Los
  valores de referencia random/experto por tarea vienen del propio D4RL
  (hoy no podemos `import d4rl` en este cluster por el tema de
  `mujoco_py`/`LD_LIBRARY_PATH` que resolvimos parcialmente en
  `eval_dt.py` — para el score normalizado alcanza con los valores
  constantes publicados, no hace falta que el import funcione).
- **Repeticiones**: seguir su convención — N episodios (ellos usan 100,
  nosotros podemos usar menos por costo, p. ej. 10-20, como el paper
  original de Decision Transformer) × 3 semillas de entrenamiento
  independientes, reportado como media ± desviación estándar entre
  semillas (no entre episodios), igual que su Ecuación 3.9.
- **Selección de checkpoint**: siempre el último del presupuesto fijado
  (sin selección por mejor validación), igual que su criterio de
  pre-registro.

### 2.5 Validación del entorno (paso previo obligatorio, análogo a su 3.1.2)
Antes de comparar DT vs HDT, replicar los números publicados del paper
original de Decision Transformer sobre estos mismos datasets D4RL con nuestra
implementación de `agent/dt.py`, y reportar el error relativo — exactamente
el mismo rol que cumple su Tabla 3.4/3.5 (confirmar que el entorno local
reproduce resultados conocidos antes de atribuir diferencias a la
topología).

**Números publicados (Chen et al. 2021, Table 2, D4RL score normalizado
100×(retorno-random)/(retorno-experto-random)):**

| Tarea | Medium | Medium-Replay | Medium-Expert |
|---|---|---|---|
| halfcheetah | 42.6 ± 0.1 | 36.6 ± 0.8 | 86.8 ± 1.3 |
| hopper | 67.6 ± 1.0 | 82.7 ± 7.0 | 107.6 ± 1.8 |
| walker2d | 74.0 ± 1.4 | 66.6 ± 3.0 | 108.1 ± 0.2 |

**Valores de referencia random/experto para el score normalizado**
(estándar D4RL, Fu et al. 2020 — resuelve la nota pendiente de §2.4):

| Tarea | Retorno random | Retorno experto |
|---|---|---|
| halfcheetah | -280.2 | 12135.0 |
| hopper | -20.3 | 3234.3 |
| walker2d | 1.6 | 4592.3 |

**⚠️ Mismatch metodológico a documentar en el informe:** el paper original
de Decision Transformer **no reporta un dataset "Expert" puro** — solo
evalúa `Medium`, `Medium-Replay` y `Medium-Expert` (esta última es una
mezcla 50/50 de trayectorias medium + expert, pensada para poner a prueba
el return-conditioning con datos de calidad mixta). Nosotros entrenamos
sobre `-expert-v2` puro (decisión ya tomada en §2, por BC puro). Además,
`medium-expert-v2` **no está descargado** en `~/.d4rl/datasets/` de este
cluster (solo hay `expert-v2`, `medium-v2`, `medium_replay-v2` por
tarea) — habría que bajarlo aparte para replicar el número exacto de la
Tabla 2.

**Segundo mismatch: hiperparámetros.** A diferencia de Benjamín (que sí
ajustó `mdp.yaml` para calzar con la Tabla 2 del paper de MaskDP, ver más
abajo), nuestro `agent/dt.yaml` NO fue armado para replicar la Tabla 9
del paper de Decision Transformer (hiperparámetros para gym/mujoco) —
se armó pensando en la paridad de parámetros DT/HDT (Etapa 0):

| Hiperparámetro | Paper DT (Tabla 9) | `dt.yaml` actual | ¿Coincide? |
|---|---|---|---|
| n_embd | 128 | 128 | sí |
| n_head | 1 | 2 | no |
| n_layer | 3 | 5 | no |
| context length (K / traj_length) | 20 | 12 | no |
| pasos de entrenamiento | 1×10⁵ | 400.010 | no (4x más) |

Consecuencia: no hay un número de Decision Transformer directamente
comparable "manzanas con manzanas" contra nuestro `-expert-v2`. Opciones,
a decidir:
1. Usar `Medium-Expert` como referencia laxa (cota superior/sanity-check,
   no objetivo exacto) — bajo BC con return-conditioning, entrenar solo
   con expert puro debería igualar o superar el score de la mezcla
   medium-expert, así que un score razonable sería "cercano o por encima"
   de esos valores, no un error relativo estricto.
2. Descargar también `medium-expert-v2` para las tres tareas y correr un
   entrenamiento adicional con ese dataset, para tener el punto de
   comparación exacto del paper (trabajo extra, un run más por tarea).
3. Documentar la diferencia metodológica explícitamente en el informe y
   no forzar una comparación numérica directa — la validación de entorno
   pasa a ser cualitativa ("¿el modelo aprende una política razonable?")
   en vez de "error relativo vs. paper".
4. Correr una validación aparte con los hiperparámetros exactos de la
   Tabla 9 (`n_head=1`, `n_layer=3`, `traj_length=20`, 100.000 pasos),
   igual que Benjamín hizo para MaskDP — un config de validación,
   *distinto* del `dt.yaml` de paridad de parámetros usado para la
   comparación DT vs HDT (Etapa 0), documentando que son dos propósitos
   distintos. Resuelve el mismatch de hiperparámetros pero no el de
   dataset (sigue faltando medium-expert-v2 para el número exacto).

*Pendiente: decidir con el usuario cuál de las tres opciones seguir antes
de armar la tabla final de comparación.*

### 2.5.1 Decisión final (2026-09-07): re-entrenar todo con dataset y config matcheados

Se investigó el código oficial de Decision Transformer
(`kzl/decision-transformer` en GitHub, `gym/experiment.py`) además del
paper, y la librería de reproducción CORL (`tinkoff-ai/CORL`), que sí
reproduce los números del paper. **Ningún benchmark público reporta el
score de DT sobre `-expert-v2` puro** — ni el paper (Tabla 2), ni CORL:
ambos solo publican `medium`, `medium-replay` y `medium-expert`. El
código oficial sí acepta `--dataset expert` como opción, pero no hay
número publicado asociado a esa corrida en ninguna fuente encontrada.

Decisión (usuario, con >1 mes de margen disponible): **matar la cola de
jobs sobre `-expert-v2` (jobs 25097-25100) y re-entrenar TODO
(halfcheetah, hopper, walker2d × DT, HDT) sobre `-medium-expert-v2`**,
que es la única variante con número citable del paper, **con los
hiperparámetros exactos del código oficial** (`gym/experiment.py`,
valores por default vía `argparse`):

| Hiperparámetro | Valor oficial | Fuente |
|---|---|---|
| n_layer | 3 | `--n_layer` default |
| n_head | 1 | `--n_head` default |
| embed_dim | 128 | `--embed_dim` default |
| dropout (embd/resid/attn) | 0.1 | `--dropout` default |
| batch_size | 64 | `--batch_size` default |
| context length K (`traj_length`) | 20 | `--K` default |
| learning_rate | 1e-4 | `--learning_rate` default (ya coincidía) |
| weight_decay | 1e-4 | `--weight_decay` default |
| warmup_steps | 10.000 | `--warmup_steps` default |
| pasos totales | 100.000 | `max_iters(10) × num_steps_per_iter(10000)` |
| max_ep_len / scale | 1000 / 1000 | hardcoded por entorno (halfcheetah/hopper/walker2d) — ya coincidía |

*(nota: `warmup_steps`/`weight_decay` del optimizador AdamW no están
implementados en nuestro `DTAgent.opt` actual — usa `torch.optim.Adam`
simple sin warmup ni weight decay. No se tocó eso en esta pasada, queda
como diferencia conocida a documentar en el informe si no se implementa
después.)*

**Decisión (2026-09-07):** no implementar todavía. Queda como el
**primer sospechoso a revisar si los resultados de los jobs 25129-25134
difieren mucho de la Tabla 2** (§2.5) una vez evaluados con `eval_dt.py`
— es la diferencia de receta de entrenamiento conocida que más falta
por cerrar (dataset y arquitectura ya matcheados). De ser necesario
implementarlo:

- Cambiar `DTAgent.opt` de `torch.optim.Adam` a `torch.optim.AdamW`,
  agrupando parámetros en dos grupos (con weight decay para matrices de
  pesos — atención, MLP, embeddings de proyección; sin weight decay para
  bias y LayerNorm, siguiendo la convención de minGPT/GPT que usa el
  código oficial de DT).
- Agregar un `torch.optim.lr_scheduler.LambdaLR` con
  `lambda step: min((step+1)/warmup_steps, 1)` (warmup lineal de 10.000
  pasos, sin decay posterior) y un `scheduler.step()` por paso en
  `update_actor`/`update`.
- Replicar el cambio en `HDTAgent.__init__` también (hoy duplica el
  `__init__` de `DTAgent` en vez de heredarlo).

**Dataset:** se descargaron `{halfcheetah,hopper,walker2d}_medium_expert-v2.hdf5`
desde el hosting oficial de D4RL
(`rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/`) y se
convirtieron con `d4rl_data.py` (mismo script, sin cambios):

| Dataset | Episodios | Ruta |
|---|---|---|
| halfcheetah-medium-expert-v2 | 2000 | `data/halfcheetah_medium_expert/halfcheetah/` |
| hopper-medium-expert-v2 | 3213 | `data/hopper_medium_expert/hopper/` |
| walker2d-medium-expert-v2 | 2190 | `data/walker2d_medium_expert/walker2d/` |

**Paridad de parámetros recalculada** (Etapa 0 rehecha con los nuevos
hiperparámetros de DT — el `n_embd=88` anterior calculado contra
`n_layer=5` ya no aplica):

| Tarea | DT (n_embd=128,n_head=1,n_layer=3) | HDT (n_embd=68,n_obs=n_act=n_top=3,n_head=1) | Desviación |
|---|---|---|---|
| halfcheetah | 727.302 | 713.734 | -1.87% |
| walker2d | 727.302 | 713.734 | -1.87% |
| hopper | 725.763 | 712.915 | -1.77% |

Config aplicada en `agent/dt.yaml` y `agent/hdt.yaml` (incluye
`batch_size=64`, dropout=0.1, `traj_length=20`, comentarios con la
fuente). Verificado con un smoke test local (5 pasos, CPU) para DT y HDT
antes de encolar — ambos corren sin errores y el conteo de parámetros
impreso coincide exactamente con el cálculo (727.302 para DT
halfcheetah).

**Jobs encolados** (reemplazan a 25097-25100, cancelados con `scancel`):

| Job ID | Corrida | Dataset |
|---|---|---|
| 25129 | DT halfcheetah | medium-expert-v2 |
| 25130 | HDT halfcheetah | medium-expert-v2 |
| 25131 | DT hopper | medium-expert-v2 |
| 25132 | HDT hopper | medium-expert-v2 |
| 25133 | DT walker2d | medium-expert-v2 |
| 25134 | HDT walker2d | medium-expert-v2 |

Scripts: `pretrain_{dt,hdt}_{halfcheetah,hopper,walker2d}_matched.sbatch`.
100.010 pasos cada uno (vs. 400.010 antes) → de referencia (§2.4.1),
deberían tardar bastante menos que los runs anteriores al ser 4x menos
pasos, aunque el batch_size también bajó de 1024 a 64 (más pasos por
segundo pero cada uno más barato — el tiempo real de wall-clock se mide
cuando terminen). Todos corren secuenciales en el mismo nodo (`hydra`).

**Nota:** los snapshots viejos (`halfcheetah_dt`, `halfcheetah_hdt`,
`hopper_dt`, etc., entrenados sobre `-expert-v2` puro con los
hiperparámetros no matcheados) NO se borraron — quedan en
`~/snapshot/{halfcheetah,hopper,walker2d}_{dt,hdt}/` por si sirven para
la opción de "validación cualitativa con expert puro" más adelante. Los
nuevos snapshots matcheados quedan en una ruta separada:
`~/snapshot/{halfcheetah,hopper,walker2d}_medium_expert_{dt,hdt}/`.

**Qué hizo Benjamín ante un problema análogo (investigado en su código,
commit `f059cb0` "Hparams changes to match paper config", rama
`hierarchical`):** ajustó `agent/mdp.yaml` para que sus hiperparámetros
coincidan exactamente con la Tabla 2 del paper original de MaskDP (Liu et
al. 2022) — `batch_size=384`, `n_embd=256`, `n_head=4`, `n_enc_layer=3`,
`n_dec_layer=2`, `traj_length=64`, comentado explícitamente como "Hparams
from paper (Table 2)". Esto le funcionó sin fricción porque el paper de
MaskDP **sí publica resultados sobre datos "expert"** (rollouts
near-expert de TD3), la misma calidad de dato que él usa — no tuvo que
resolver un mismatch de calidad de dataset como el nuestro. La diferencia
es de los papers de origen, no de su método: el paper de Decision
Transformer directamente no reporta una variante "Expert" pura para
D4RL gym-mujoco, así que copiar su enfoque (matchear hiperparámetros y
comparar con la misma calidad de dato que usa el paper) no alcanza para
cerrar este punto — el número que haría falta no existe en la fuente.

### 2.4.1 Entrenamiento en hopper y walker2d — lanzado

Respuesta a la pregunta "¿hace falta correr DT y HDT en hopper/walker2d,
o alcanza con HDT?": sí, hace falta correr los dos. El objetivo del
trabajo es comparar la topología DT vs HDT, y además la validación de
entorno (§2.5) exige replicar los números publicados de Decision
Transformer en cada dataset con nuestra implementación de `dt.py` antes
de atribuir diferencias a la topología — sin una corrida de DT en
hopper/walker2d no hay con qué comparar ni validar el entorno para esas
tareas. Solo entrenar HDT hubiera dejado esas dos tareas sin línea base.

Se crearon 4 sbatch nuevos, mismo patrón que
`pretrain_dt_full.sbatch`/`pretrain_hdt_full.sbatch` (los que ya se usaron
para halfcheetah, jobs 23647/24568 dt y 24531/24569 hdt), cambiando
`task`/`obs_dim`/`action_dim`/rutas de datos y snapshot:

- `pretrain_dt_hopper.sbatch` / `pretrain_hdt_hopper.sbatch`
  (obs_dim=11, action_dim=3)
- `pretrain_dt_walker2d.sbatch` / `pretrain_hdt_walker2d.sbatch`
  (obs_dim=17, action_dim=6)

Config: mismo `agent/dt.yaml`/`agent/hdt.yaml` (este último ya con
`n_embd=88` de la paridad de parámetros, §1.1), 400.010 pasos, seed=1
(default), `use_wandb=false`.

Encolados con `sbatch` el 2026-09-07:

| Job ID | Corrida | Estado al encolar |
|---|---|---|
| 25097 | dt hopper | pendiente (Resources) |
| 25098 | hdt hopper | pendiente (ReqNodeNotAvail) |
| 25099 | dt walker2d | pendiente (ReqNodeNotAvail) |
| 25100 | hdt walker2d | pendiente (ReqNodeNotAvail) |

Los 4 jobs están pineados a `--nodelist=hydra` (un solo nodo), así que
corren uno detrás del otro, no en paralelo — mismo comportamiento que
tuvieron los jobs de halfcheetah.

**Duración de referencia** (halfcheetah, 400.010 pasos, `sacct`):

| Job | Corrida | Duración |
|---|---|---|
| 23647 | DT full (1er intento, 31/08) | 12h 28m |
| 24568 | DT full (definitivo, 04-05/09) | 7h 30m |
| 24569 | HDT full (definitivo, 05/09) | 8h 51m |

Los definitivos corrieron secuenciales en el mismo nodo: ~16h 21m el par
completo. Con esa referencia, los 4 jobs de hopper/walker2d (25097-25100,
también secuenciales en el mismo nodo) deberían tardar en total del orden
de **30-35h** desde que el nodo se libere. Snapshots quedan en
`~/snapshot/hopper_dt/`, `~/snapshot/hopper_hdt/`,
`~/snapshot/walker2d_dt/`, `~/snapshot/walker2d_hdt/`.

*Nota: entrenados con `n_embd=88` para HDT (post-paridad de parámetros).
Los snapshots de halfcheetah ya entrenados (jobs 23647/24568/24531/24569)
son previos a este cambio de `n_embd`, es decir el HDT de halfcheetah
quedó con `n_embd=128` (2.172.934 params) y NO tiene paridad de
parámetros con su DT. Si se quiere una comparación con paridad estricta
en las tres tareas, habría que re-entrenar HDT en halfcheetah también con
`n_embd=88` (pendiente de decisión).*

### 2.4.2 `eval_dt.py` generalizado a las 3 tareas (mientras se esperaban los jobs)

`eval_dt.py` original solo soportaba halfcheetah (obs/action dims y
`gym.make("HalfCheetah-v2")` hardcodeados), corría un único episodio y
grababa siempre video — no servía para el protocolo de §2.4 (N episodios,
score normalizado D4RL, 3 tareas). Se generalizó mientras se esperaba que
arrancaran los jobs 25129-25134:

- Diccionario `TASKS` con, por tarea: `env_id` (confirmado que
  `HalfCheetah-v2`/`Hopper-v2`/`Walker2d-v2` son los IDs correctos para
  el gym+mujoco_py de este cluster, con las dims obs/action esperadas),
  `obs_dim`/`action_dim`, los targets de retorno de evaluación oficiales
  de `kzl/decision-transformer` (`env_targets` en `gym/experiment.py`:
  halfcheetah [12000, 6000], hopper [3600, 1800], walker2d [5000, 2500]),
  y los retornos random/experto de D4RL para el score normalizado (ya
  citados en §2.5).
- `--task {halfcheetah,hopper,walker2d}` selecciona todo lo anterior.
  `--target-return` sigue siendo overrideable a mano (default: el primer
  target oficial de la tarea).
- `--num-episodes` corre N episodios y reporta retorno crudo y score
  normalizado D4RL como media ± desviación estándar (antes: un solo
  episodio, sin normalizar).
- Grabar video ahora es opcional (`--save-video`, solo graba el primer
  episodio) en vez de obligatorio en cada corrida — evita el costo de
  encodear video en los N episodios cuando solo se necesita el número.

Verificado con snapshots dummy (pesos random, sin entrenar) para las 4
combinaciones tarea×agente relevantes: conteo de parámetros impreso
coincide exactamente con lo calculado en §2.5.1 (hopper DT: 725.763,
walker2d HDT: 713.734), rollout corre, video opcional funciona. Falta
correrlo contra checkpoints reales una vez terminen los jobs.

### 0.2 Hallazgo: checkpoints reales de Benjamín accesibles en el cluster (2026-09-07)

Investigando la Etapa 2 (ver §3) se encontró que `/home/bmancilla/archive/MaskDP/`
tiene permisos de lectura abiertos y contiene, además del código ya
descrito en §0.1:

- `maskdp_snapshots/`: cientos de carpetas de experimentos propioceptivos
  (DMC) ya entrenados, con `snapshot_*.pt` reales y un `wandb_info.txt`
  por corrida (Run ID/URL del proyecto wandb
  `benjamin-mancilla-universidad-de-chile/maskdp-mm`). Nombres de carpeta
  distinguen `self` (unistream) vs `cross`/`neck180_cross` (jerárquico,
  atención cruzada) — la Figura 3.1 de su metodología, ya entrenada.
- `maskdp_snapshots_procgen/`: lo mismo para la Etapa 2 (CoinRun/Procgen),
  wandb project `maskdp-mm-procgen`.
- `pretrained_encoders/procgen_coinrun_easy_encoder.pt`: encoder IMPALA
  congelado ya entrenado, listo para usar.
- `v-d4rl/`: repo de V-D4RL con datos propios en `drqv2_data/`.

**Decisión del usuario:** no hace falta tocar nada de esto. Benjamín ya
terminó su informe y el usuario lo tiene — la comparación numérica contra
sus resultados se hace al final del proyecto (cuando el usuario escriba
su propio informe), usando los números ya publicados en el informe de
Benjamín, no re-evaluando sus checkpoints en vivo. Lo único que hace
falta de acá en adelante es **dejar bien guardados y documentados
nuestros propios resultados** a medida que se generan (que es
justamente el propósito de este documento). No se tocaron ni se van a
tocar los checkpoints/código de Benjamín salvo que el usuario lo pida
explícitamente más adelante.

### 2.6 Resultados — primera tanda, sin AdamW/warmup (jobs 25129-25134, 2026-09-08)

Esta primera tanda de entrenamiento (§2.5.1) todavía usaba
`torch.optim.Adam` simple, sin weight decay agrupado ni warmup lineal —
la diferencia de receta pendiente frente al código oficial de Decision
Transformer que ya se había señalado como sospechosa antes de correr
estos jobs. Los resultados de acá quedan como la línea base "sin AdamW";
la comparación contra la segunda tanda "con AdamW" está en §2.7.

Los 6 jobs (§2.5.1) terminaron sin errores (`sacct`: todos `COMPLETED`,
`ExitCode 0:0`, sin `nan`/`Traceback` en los `.err`):

| Job ID | Corrida | Duración |
|---|---|---|
| 25129 | DT halfcheetah | 0:58:07 |
| 25130 | HDT halfcheetah | 1:53:13 |
| 25131 | DT hopper | 0:57:22 |
| 25132 | HDT hopper | 1:56:23 |
| 25133 | DT walker2d | 0:54:40 |
| 25134 | HDT walker2d | 1:57:41 |

Evaluados con `eval_dt.py` sobre el checkpoint final (`snapshot_100000.pt`,
env `dt-env`), 10 episodios, `target_return` = primer target oficial de
`kzl/decision-transformer` por tarea (§2.4.2), video del primer episodio
grabado con `--save-video`:

| Tarea | DT retorno | DT score D4RL | HDT retorno | HDT score D4RL | Referencia paper (Medium-Expert) |
|---|---|---|---|---|---|
| halfcheetah | -64.89 ± 1.22 | **1.73 ± 0.01** | 10948.11 ± 166.05 | **90.44 ± 1.34** | 86.8 ± 1.3 |
| hopper | 1615.88 ± 137.22 | **50.27 ± 4.22** | 3072.72 ± 852.14 | **95.04 ± 26.18** | 107.6 ± 1.8 |
| walker2d | 3529.82 ± 926.31 | **76.86 ± 20.18** | 3480.10 ± 699.26 | **75.77 ± 15.23** | 108.1 ± 0.2 |

**Videos generados** (primer episodio de cada evaluación, `videos/`):
`halfcheetah_dt.mp4`, `halfcheetah_hdt.mp4`, `hopper_dt.mp4`,
`hopper_hdt.mp4`, `walker2d_dt.mp4`, `walker2d_hdt.mp4` (los `*_400000.mp4`
preexistentes en la misma carpeta son de los runs viejos sobre
`-expert-v2`, no de estos jobs — se dejaron sin tocar).

**⚠️ Hallazgo a investigar — DT en halfcheetah colapsó:** DT retorna
score ~1.73 (retorno negativo, peor que política aleatoria en la práctica)
mientras que HDT en la misma tarea y el mismo dataset llega a 90.44,
por encima del valor publicado. En hopper y walker2d, en cambio, DT sí
aprende una política razonable (50.27 y 76.86) aunque por debajo de HDT y
del paper. Esto no es el patrón esperado (HDT sistemáticamente mejor que
DT en TODAS las tareas, y DT colapsando solo en una) — más bien sugiere
que la corrida de DT halfcheetah (job 25129) tuvo un problema puntual
(inicialización, algún desbalance del batch de 64, o la ausencia de
warmup/AdamW pesando más en esta tarea en particular) en vez de una
diferencia real de capacidad entre topologías. **Antes de reportar estos
números como comparación final DT vs HDT, hay que**:
1. Repetir DT halfcheetah con al menos otra semilla para descartar que sea
   una corrida puntual mala.
2. ~~Revisar la curva de `BR` (behavior/return loss) de wandb del job
   25129 contra 25131/25133 (DT hopper/walker2d) para ver si divergió o
   quedó estancada.~~ Hecho (2026-09-08), ver diagnóstico abajo — **no
   explica el colapso**.
3. Si se confirma que persiste, es el candidato más fuerte para justificar
   implementar el AdamW + warmup lineal pendiente (§2.5.1, hiperparámetro
   no matcheado con el código oficial de DT), ya que el resto de la receta
   (dataset, arquitectura, batch, dropout) ya está igualada al paper.

**Diagnóstico (2026-09-08): la curva de entrenamiento NO muestra nada
anómalo.** Corrección primero: `BR` en la salida de consola/`sacct` es
`batch_reward` (una estadística del batch muestreado — la recompensa
promedio de los datos, no una pérdida), por eso se ve plana (~7.77) en
los tres `.out` — es esperable, no informa nada sobre el entrenamiento.
La métrica real de pérdida es `action_loss` (MSE actor, `DTAgent.
update_actor`), que sí se loguea (a `train.csv` y TensorBoard en cada
corrida, vía `Logger`, aunque no se imprime en consola porque no está en
`TRAIN_FORMAT` de `logger.py`) pero no se había revisado hasta ahora.
Leyendo `train.csv` de las 4 corridas relevantes
(`models/{halfcheetah,hopper,walker2d}_medium_expert_dt/.../train.csv` y
`models/halfcheetah_medium_expert_hdt/.../train.csv`):

| Corrida | `action_loss` final (paso 100.000) | Forma de la curva |
|---|---|---|
| DT halfcheetah (job 25129) | **0.0285** | Monótona decreciente, sin nan/spikes, converge más rápido que las otras |
| DT hopper (job 25131) | 0.0554 | Igual de estable |
| DT walker2d (job 25133) | 0.0311 | Igual de estable |
| HDT halfcheetah (job 25130) | 0.0299 | Igual de estable |

DT halfcheetah entrena **igual de bien o mejor** (menor MSE final) que
las otras tres corridas — no hay divergencia, estancamiento ni
inestabilidad numérica que explique el colapso en evaluación. Es más:
tiene una pérdida de entrenamiento **menor** que HDT en la misma tarea,
pero un score de evaluación catastróficamente peor (1.73 vs 90.44). Esto
descarta con bastante confianza que la causa sea el optimizador (Adam
simple vs. AdamW+warmup) — un problema de optimización debería verse
reflejado en la curva de pérdida, y acá no aparece. El problema está en
otro lado: la política aprende bien a imitar acción-a-acción en el
dataset (evaluación *open-loop*/teacher-forced, que es lo que mide
`action_loss`), pero falla en producir un buen rollout *closed-loop* (lo
que mide `eval_dt.py`) — es decir, un caso de **exposure bias /
error compuesto**, agravado en halfcheetah porque el entorno no tiene
condición de término temprano (corre los 1000 pasos completos siempre,
a diferencia de hopper/walker2d que terminan al caerse): en los rollouts
de DT halfcheetah los 10 episodios corrieron los 1000 pasos completos
pero con retorno negativo, consistente con una política que sostiene un
error de fase/timing durante todo el episodio en vez de recuperarse o
terminar temprano como en hopper/walker2d.

**Conclusión para la comparación con AdamW (§2.7):** dado este
diagnóstico, es poco probable que la tanda con AdamW+warmup (jobs
25393-25398, todavía corriendo) resuelva el colapso de DT en halfcheetah
— el mecanismo sospechado ahora es específico de la arquitectura (una
sola secuencia de auto-atención con `n_head=1` sosteniendo mal la fase
de un gait cíclico) o de la evaluación closed-loop, no del optimizador.
Igual vale la pena esperar el resultado (ya está corriendo, sin costo
adicional) para confirmar o descartar esta hipótesis empíricamente antes
de invertir tiempo en depurar la arquitectura o el protocolo de
evaluación.

Con la salvedad anterior, panorama general: los scores de HDT (90.44,
95.04, 75.77) están en el rango de lo publicado para Medium-Expert
(86.8, 107.6, 108.1) — razonablemente cerca en halfcheetah/hopper, algo
por debajo en walker2d — así que **HDT sí replica el orden de magnitud del
paper**, lo cual valida el entorno (§2.5) al menos para esa topología.

### 2.7 Segunda tanda, con AdamW + warmup (jobs 25393-25398, 2026-09-08)

Dado el colapso de DT en halfcheetah (§2.6, con Adam simple), se decidió
implementar de una vez el hiperparámetro pendiente desde §2.5.1 (AdamW con
weight decay agrupado + warmup lineal, igual al código oficial de
`kzl/decision-transformer`) en vez de asumir que es la causa sin probarlo,
y correr una segunda tanda completa (mismas 3 tareas × DT/HDT) para poder
comparar directamente "sin AdamW" (§2.6) vs. "con AdamW" (acá) con el
resto de la receta idéntica.

**Cambios de código:**
- `agent/dt.py`: nueva función `configure_optimizer(model, lr,
  weight_decay, warmup_steps)`, compartida por `DTAgent` y `HDTAgent`.
  Agrupa parámetros igual que minGPT: `nn.Linear.weight` (atención, MLP,
  proyecciones de entrada) va con weight decay; bias, `nn.LayerNorm` y
  `nn.Embedding` (pos_embed) sin weight decay. Devuelve
  `(AdamW(...), LambdaLR(warmup lineal sin decay posterior))`.
- `DTAgent.__init__`/`HDTAgent.__init__`: reemplazan
  `torch.optim.Adam(self.model.parameters(), lr=lr)` por
  `self.opt, self.scheduler = configure_optimizer(...)`, con
  `weight_decay=1e-4`/`warmup_steps=10000` como nuevos parámetros
  (default = valores oficiales, para no romper otros llamadores como
  `eval_dt.py`).
- `DTAgent.update_actor`: agrega `self.scheduler.step()` después de
  `self.opt.step()`. `HDTAgent` lo hereda sin cambios (no sobreescribe
  `update_actor`).
- `agent/dt.yaml`/`agent/hdt.yaml`: agregan `weight_decay: 1e-4` y
  `warmup_steps: 10000`, documentados como matcheados al código oficial.

**Verificación antes de encolar:**
- `test_dt.py`/`test_hdt.py` (smoke tests standalone existentes): todos
  los checks pasan sin cambios, incluyendo entrenamiento con batch real
  de D4RL (loss finito) — confirma que el cambio de optimizador no rompe
  el forward/backward ni el conteo de parámetros (727.302 DT / 713.734
  HDT en halfcheetah, igual que §2.5.1).
- Smoke test de `pretrain_dt.py` en CPU, 5 pasos, `agent=dt` y `agent=hdt`
  sobre halfcheetah: corre sin errores con el nuevo optimizador.

**Snapshots viejos preservados:** los jobs 25129-25134 (§2.5.1/§2.6, Adam
simple sin warmup) **no se tocaron** — siguen en
`~/snapshot/{halfcheetah,hopper,walker2d}_medium_expert_{dt,hdt}/`. Los
sbatch nuevos (`pretrain_{dt,hdt}_{halfcheetah,hopper,walker2d}_matched_adamw.sbatch`,
copias de los `_matched.sbatch` originales) escriben a rutas separadas con
sufijo `_adamw` (`snapshot_dir`, `hydra.run.dir` y `exp_name`), verificado
que no colisionan antes de encolar:

| Job ID | Corrida | Snapshot dir |
|---|---|---|
| 25393 | DT halfcheetah (AdamW+warmup) | `~/snapshot/halfcheetah_medium_expert_dt_adamw/` |
| 25394 | HDT halfcheetah (AdamW+warmup) | `~/snapshot/halfcheetah_medium_expert_hdt_adamw/` |
| 25395 | DT hopper (AdamW+warmup) | `~/snapshot/hopper_medium_expert_dt_adamw/` |
| 25396 | HDT hopper (AdamW+warmup) | `~/snapshot/hopper_medium_expert_hdt_adamw/` |
| 25397 | DT walker2d (AdamW+warmup) | `~/snapshot/walker2d_medium_expert_dt_adamw/` |
| 25398 | HDT walker2d (AdamW+warmup) | `~/snapshot/walker2d_medium_expert_hdt_adamw/` |

Encolados el 2026-09-08, mismo patrón que §2.5.1 (todos pineados a
`--nodelist=hydra`, corren secuenciales, ~1-2h cada uno según duración de
referencia). Misma config de resto de hiperparámetros (dataset
medium-expert-v2, `n_embd`/`n_layer`/`traj_length`/`batch_size` sin
cambios) — el único cambio metodológico entre esta tanda y la de
§2.5.1/§2.6 es el optimizador, para poder aislar su efecto.

**Pendiente una vez terminen:** evaluar con `eval_dt.py` igual que §2.6 y
comparar ambas tandas (Adam simple vs. AdamW+warmup) para las 3 tareas —
en particular confirmar si DT halfcheetah deja de colapsar.

**⚠️ Actualización (2026-09-08): jobs 25393-25398 cancelados antes de
terminar** (con `scancel`, ninguno alcanzó a generar `snapshot_100000.pt`
— sin pérdida real, el más avanzado llevaba 15 min de 100.010 pasos).
Motivo: mientras corrían, se encontró un segundo problema más grave en el
pipeline (§2.8, falta de normalización de observaciones) que hay que
arreglar en el mismo re-entrenamiento en vez de encolarlo por separado —
ver §2.8 para la tanda que reemplaza a esta.

### 2.8 Hallazgo — falta normalización de observaciones (2026-09-08)

Con la curva de entrenamiento de DT halfcheetah descartada como causa
(§2.6, converge normal), se buscó qué otra parte de la receta del código
oficial de Decision Transformer no estuviera matcheada más allá de los
hiperparámetros de la Tabla 9. Se encontró que el código oficial
(`kzl/decision-transformer`, `gym/experiment.py`) normaliza las
observaciones con z-score (`(obs - state_mean) / state_std`, media/std
calculadas sobre todo el dataset de entrenamiento) **tanto al armar los
batches de entrenamiento como en el rollout de evaluación** (usando el
mismo `state_mean`/`state_std` guardado) — un paso de preprocesamiento de
datos, no un hiperparámetro de la Tabla 9, así que quedó fuera de todo el
matching hecho hasta ahora (§2.5.1).

Se confirmó que nuestro pipeline no lo hacía en ningún punto: `d4rl_data.py`
guarda las observaciones crudas del HDF5 sin transformar,
`OfflineReplayBuffer._sample` (`replay_buffer.py`) las entrega tal cual, y
`DecisionTransformer`/`HierarchicalDecisionTransformer` las pasan directo
a la proyección lineal de estado sin normalizar; `eval_dt.py` tampoco
normalizaba en el rollout.

**Por qué esto explica mejor el patrón observado que el optimizador:**
las dimensiones de observación de MuJoCo tienen escalas muy distintas
entre sí (posiciones, velocidades, ángulos). Sin normalizar, una
proyección lineal chica (`n_embd` 128/68, 3 capas, 100k pasos) tiene que
aprender esa recalibración con presupuesto limitado — consistente con que
**las tres tareas, no solo halfcheetah, quedaron por debajo del paper**
(§2.6). Y explica el colapso específico de DT en halfcheetah mejor que
cualquier hipótesis de optimizador: es la tarea con rango de velocidades
más disparejo entre dimensiones, y al no terminar nunca antes de los 1000
pasos (a diferencia de hopper/walker2d, que terminan al caerse), un error
de escala/timing se sostiene y compone durante todo el episodio en vez de
cortarse temprano.

**Cambios de código:**
- `agent/dt.py` (`DecisionTransformer`) y `agent/hdt.py`
  (`HierarchicalDecisionTransformer`): agregan buffers `obs_mean`/
  `obs_std` (shape `obs_dim`, default identidad `mean=0, std=1`) y un
  método `set_obs_stats(mean, std)`. `forward()` normaliza
  `obs = (obs - obs_mean) / obs_std` antes de la proyección de estado
  (`state_embed` en DT, `obs_encoder` en HDT). Al ser buffers (no
  parámetros), quedan incluidos automáticamente en `model.state_dict()`
  → se guardan en el checkpoint y viajan con él.
- `DTAgent.__init__`/`HDTAgent.__init__`: nuevos parámetros
  `obs_mean=None, obs_std=None`; si se pasan, llaman a
  `self.model.set_obs_stats(...)` después de crear el modelo.
- `pretrain_dt.py`: nueva función `compute_obs_stats(replay_dir, domain)`
  que calcula media/std sobre todas las observaciones de los episodios
  `.npz` de entrenamiento (excluyendo el último estado de *bootstrap*).
  Se llama antes de instanciar el agente y el resultado se pasa como
  `obs_mean`/`obs_std` a `hydra.utils.instantiate(cfg.agent, ...)`.
- **Retrocompatibilidad con checkpoints viejos:** `eval_dt.py` y el
  `resume` de `pretrain_dt.py` ahora cargan el checkpoint con
  `load_state_dict(..., strict=False)` — los checkpoints de §2.5.1/§2.6/
  §2.7 (previos a este cambio) no tienen los buffers `obs_mean`/
  `obs_std` en su `state_dict`, así que quedan en su default de
  identidad, que es exactamente cómo se entrenaron. Verificado
  reevaluando `halfcheetah_medium_expert_hdt/snapshot_100000.pt` (job
  25130): mismo retorno por episodio que en §2.6 (10790.26, 10878.64),
  confirma que no cambió nada para checkpoints viejos.
- **Flag adicional `use_adamw`** (`DTAgent`/`HDTAgent`, default `True`):
  para poder correr la comparación "con/sin AdamW" pedida por el usuario
  manteniendo la normalización fija en ambas ramas, en vez de solo
  suponer su efecto. `use_adamw=False` vuelve a `torch.optim.Adam` simple
  (sin weight decay agrupado ni warmup; scheduler no-op para no
  duplicar lógica en `update_actor`). Agregado a `dt.yaml`/`hdt.yaml`
  como `use_adamw: true`, overrideable por línea de comandos
  (`agent.use_adamw=false`).

**Verificación antes de encolar:**
- `test_dt.py`/`test_hdt.py`: todos los checks pasan sin cambios (la
  normalización es identidad por default, no altera el comportamiento
  numérico de los tests existentes que no pasan `obs_mean`/`obs_std`).
- Smoke test de `pretrain_dt.py` en CPU (5 pasos, halfcheetah, `agent=dt`
  y `agent=hdt`, con `agent.use_adamw=false` en uno de los dos): corre
  sin errores, `compute_obs_stats` calcula medias/std con la variación
  esperada entre dimensiones (ej. la dimensión de altura del torso salió
  con media ~8.1 y std ~3.4, muy distinta a otras dimensiones con media
  ~0 — justamente el tipo de disparidad de escala que la normalización
  corrige), y el checkpoint guardado (`snapshot_0.pt`) contiene los
  buffers `obs_mean`/`obs_std` con esos valores reales. Conteo de
  parámetros entrenables sin cambios (727.302 DT / 713.734 HDT
  halfcheetah) — los buffers no son parámetros.

**Jobs encolados** (reemplazan a 25393-25398, §2.7): 12 en total —
3 tareas × 2 agentes (DT/HDT) × 2 optimizadores (Adam simple / AdamW +
warmup), todos con la normalización de obs ya activa en ambas ramas, para
aislar el efecto de AdamW por separado del de la normalización:

| Job ID | Corrida | Snapshot dir |
|---|---|---|
| 25406 | DT halfcheetah, Adam simple + norm | `~/snapshot/halfcheetah_medium_expert_dt_norm_adam/` |
| 25407 | DT halfcheetah, AdamW+warmup + norm | `~/snapshot/halfcheetah_medium_expert_dt_norm_adamw/` |
| 25408 | HDT halfcheetah, Adam simple + norm | `~/snapshot/halfcheetah_medium_expert_hdt_norm_adam/` |
| 25409 | HDT halfcheetah, AdamW+warmup + norm | `~/snapshot/halfcheetah_medium_expert_hdt_norm_adamw/` |
| 25410 | DT hopper, Adam simple + norm | `~/snapshot/hopper_medium_expert_dt_norm_adam/` |
| 25411 | DT hopper, AdamW+warmup + norm | `~/snapshot/hopper_medium_expert_dt_norm_adamw/` |
| 25412 | HDT hopper, Adam simple + norm | `~/snapshot/hopper_medium_expert_hdt_norm_adam/` |
| 25413 | HDT hopper, AdamW+warmup + norm | `~/snapshot/hopper_medium_expert_hdt_norm_adamw/` |
| 25414 | DT walker2d, Adam simple + norm | `~/snapshot/walker2d_medium_expert_dt_norm_adam/` |
| 25415 | DT walker2d, AdamW+warmup + norm | `~/snapshot/walker2d_medium_expert_dt_norm_adamw/` |
| 25416 | HDT walker2d, Adam simple + norm | `~/snapshot/walker2d_medium_expert_hdt_norm_adam/` |
| 25417 | HDT walker2d, AdamW+warmup + norm | `~/snapshot/walker2d_medium_expert_hdt_norm_adamw/` |

Scripts: `pretrain_{dt,hdt}_{halfcheetah,hopper,walker2d}_matched_norm_{adam,adamw}.sbatch`,
copias de los `_matched.sbatch` originales de §2.5.1 con `snapshot_dir`/
`hydra.run.dir`/`exp_name` en rutas separadas (sufijo `_norm_adam` o
`_norm_adamw`, verificado que no colisionan con nada existente antes de
encolar) y `agent.use_adamw=false`/`true` según corresponda. Encolados el
2026-09-08, todos pineados a `--nodelist=hydra` (corren secuenciales — al
ser 12 en vez de 6, del orden de 12-24h en total en vez de 8-12h).

**Snapshots viejos preservados, ninguno tocado:** ni los de §2.5.1/§2.6
(`~/snapshot/{halfcheetah,hopper,walker2d}_medium_expert_{dt,hdt}/`, Adam
simple sin normalizar) ni ningún otro — todos en rutas nuevas con sufijo
`_norm_adam`/`_norm_adamw`.

**Pendiente una vez terminen:** evaluar los 12 con `eval_dt.py` (ya
retrocompatible) y armar una tabla de 3 filas × 4 columnas (sin
AdamW/sin norm de §2.6, con AdamW/sin norm si se repite, sin AdamW/con
norm, con AdamW/con norm) para aislar el efecto de cada cambio por
separado — en particular, la hipótesis de §2.6 predice que la
normalización por sí sola (sin AdamW) ya debería destrabar el colapso de
DT en halfcheetah, y que AdamW aportaría poco o nada encima de eso.

### 2.9 Resultados — tanda con normalización de observaciones (jobs 25406-25417, 2026-09-15)

Los 12 jobs de §2.8 (3 tareas × {DT, HDT} × {Adam simple, AdamW+warmup},
todos con `obs_mean`/`obs_std` ya activos) terminaron `COMPLETED` sin
errores (`sacct`, `ExitCode 0:0` los 12). Evaluados con `eval_dt.py` sobre
el checkpoint final (`snapshot_100000.pt`), 10 episodios cada uno, mismo
protocolo que §2.6 (`target_return` = primer target oficial de
`kzl/decision-transformer` por tarea, video del primer episodio con
`--save-video`). Logs en `eval_results/*_norm_{adam,adamw}.log`, videos en
`videos/*_norm_{adam,adamw}.mp4`.

**Duración de entrenamiento:**

| Job ID | Corrida | Duración |
|---|---|---|
| 25406 | DT halfcheetah, Adam simple + norm | 0:55:53 |
| 25407 | DT halfcheetah, AdamW+warmup + norm | 0:56:14 |
| 25408 | HDT halfcheetah, Adam simple + norm | 1:54:49 |
| 25409 | HDT halfcheetah, AdamW+warmup + norm | 1:55:17 |
| 25410 | DT hopper, Adam simple + norm | 0:56:23 |
| 25411 | DT hopper, AdamW+warmup + norm | 0:56:24 |
| 25412 | HDT hopper, Adam simple + norm | 1:53:03 |
| 25413 | HDT hopper, AdamW+warmup + norm | 1:52:53 |
| 25414 | DT walker2d, Adam simple + norm | 1:00:59 |
| 25415 | DT walker2d, AdamW+warmup + norm | 0:56:46 |
| 25416 | HDT walker2d, Adam simple + norm | 1:53:15 |
| 25417 | HDT walker2d, AdamW+warmup + norm | 1:54:34 |

**Score normalizado D4RL (media ± desviación estándar, 10 episodios):**

| Tarea | Agente | Adam simple + norm | AdamW+warmup + norm | Referencia paper (Medium-Expert) |
|---|---|---|---|---|
| halfcheetah | DT | **89.01 ± 4.49** | 71.52 ± 34.51 | 86.8 ± 1.3 |
| halfcheetah | HDT | 61.41 ± 24.31 | 71.09 ± 23.51 | 86.8 ± 1.3 |
| hopper | DT | 52.18 ± 5.66 | 60.13 ± 15.81 | 107.6 ± 1.8 |
| hopper | HDT | 54.15 ± 7.11 | 54.67 ± 9.00 | 107.6 ± 1.8 |
| walker2d | DT | 73.74 ± 20.87 | 76.58 ± 12.53 | 108.1 ± 0.2 |
| walker2d | HDT | 76.69 ± 10.45 | 76.36 ± 11.41 | 108.1 ± 0.2 |

**Comparación contra la tanda sin normalizar (§2.6, Adam simple, sin
`obs_mean`/`obs_std`) — aísla el efecto de la normalización:**

| Tarea | Agente | Sin norm (§2.6) | Con norm, Adam simple (§2.9) | Δ |
|---|---|---|---|---|
| halfcheetah | DT | 1.73 ± 0.01 | 89.01 ± 4.49 | **+87.28** |
| halfcheetah | HDT | 90.44 ± 1.34 | 61.41 ± 24.31 | -29.03 |
| hopper | DT | 50.27 ± 4.22 | 52.18 ± 5.66 | +1.91 |
| hopper | HDT | 95.04 ± 26.18 | 54.15 ± 7.11 | -40.89 |
| walker2d | DT | 76.86 ± 20.18 | 73.74 ± 20.87 | -3.12 |
| walker2d | HDT | 75.77 ± 15.23 | 76.69 ± 10.45 | +0.92 |

**Lectura de los resultados:**

1. **Confirma la hipótesis de §2.8 para el caso que la motivó**: el
   colapso de DT en halfcheetah desaparece por completo al normalizar
   observaciones (score 1.73 → 89.01, con Adam simple, sin tocar el
   optimizador) — queda al nivel del paper (86.8) y es el mejor resultado
   de las 4 combinaciones para esa tarea. Esto aísla la causa: era la
   falta de normalización, no el optimizador, exactamente como predijo el
   diagnóstico de la curva de pérdida en §2.6.
2. **AdamW+warmup no aporta de forma consistente una vez que ya hay
   normalización** — mejora HDT halfcheetah (+9.7) y DT hopper (+7.9),
   pero empeora bastante DT halfcheetah (-17.5, con una desviación
   estándar muy alta: 34.51, es decir episodios sueltos con retorno casi
   nulo entre episodios con retorno pleno) y es prácticamente neutro en
   walker2d (ambos agentes) y HDT hopper. No hay un patrón de "AdamW
   siempre ayuda" — su efecto parece dominado por varianza entre semillas
   más que por una mejora sistemática, consistente con la predicción de
   §2.6 de que el optimizador no era el mecanismo principal.
3. **HDT no domina sistemáticamente a DT en esta tanda**, a diferencia de
   §2.6: con normalización, DT iguala o supera a HDT en halfcheetah (Adam
   simple) y hopper (ambos optimizadores), y quedan prácticamente
   empatados en walker2d. La caída de HDT en halfcheetah (90.44 → 61.41
   con Adam simple) es inesperada — normalizar debería ayudar o ser
   neutro, no empeorar; candidato a revisar: una sola corrida por celda
   (sin múltiples semillas todavía, ver punto 4) hace que no se pueda
   distinguir varianza de semilla de un efecto real de la normalización
   sobre la topología jerárquica.
4. **Ninguna combinación alcanza el número del paper en hopper/walker2d**
   (mejor caso: DT hopper AdamW 60.13 vs. 107.6 del paper; HDT walker2d
   Adam 76.69 vs. 108.1) — a diferencia de halfcheetah DT, que sí lo
   alcanza. Pendiente investigar si es un problema específico de esas dos
   tareas (terminan episodio temprano al caerse, a diferencia de
   halfcheetah) o si hace falta más presupuesto de entrenamiento.
5. **Ninguna de estas corridas tiene todavía múltiples semillas** — el
   protocolo de §2.4 pide media±std entre semillas, no entre episodios de
   una sola semilla (que es lo que se reporta acá, igual que en §2.6).
   Antes de sacar conclusiones firmes sobre DT vs. HDT o sobre el efecto
   de AdamW hace falta repetir con ≥2 semillas adicionales por celda,
   sobre todo dado el tamaño de algunas desviaciones estándar entre
   episodios (p. ej. DT halfcheetah AdamW: 34.51).

**Pendiente:** correr semillas adicionales (punto 5) antes de fijar una
conclusión DT vs. HDT o Adam vs. AdamW; decidir si se sigue extendiendo el
entrenamiento en hopper/walker2d (punto 4) para acercarse más al paper.

### 2.10 Hallazgo — el return-to-go de entrenamiento estaba mal calculado (2026-09-21)

Al revisar por qué hopper/walker2d quedaban 30-55 puntos bajo el paper (§2.9,
punto 4) se comparó nuestro pipeline contra el código oficial de
`kzl/decision-transformer` (`gym/experiment.py`, `seq_trainer.py`,
consultados en GitHub el 2026-09-21). Dos diferencias en el return-to-go
(rtg), la señal que define a Decision Transformer:

1. **Ventana en vez de episodio.** `DTAgent.compute_returns_to_go` recibía
   solo los K=20 pasos del batch, así que el rtg era la suma de rewards
   dentro de la ventana y no hasta el final del episodio. El oficial usa
   `discount_cumsum(traj['rewards'][si:], gamma=1.)`.
2. **Descuento 0.99.** El `discount` del buffer (`pretrain_dt.yaml`,
   `discount: 0.99`) es un default heredado de los agentes RL de MaskDP
   (target del crítico), nunca una decisión para DT. El oficial usa
   `gamma=1.` (sin descontar). Confirmado: no hay nada sobre descuento en
   este documento.

**Medición sobre hopper medium-expert (datos reales):** el rtg visto en
entrenamiento llegaba como máximo a 0.099 (escalado ÷1000, media 0.058),
mientras el retorno real de un episodio es 331-3759 (media 2109) y el
`target_return` de evaluación es 3600 (3.6 escalado). El modelo nunca vio
un rtg mayor a 0.1 y en evaluación se lo pedíamos 36 veces más grande; en
entrenamiento tampoco distinguía episodios buenos de malos. Efectivamente
era clonación de comportamiento sobre una mezcla medium+expert, lo que
explica hopper ≈ 52-60 y que en §4.3 "return" casi no pesara. Afecta a
TODOS los resultados de D4RL (DT y HDT, §2.6-§2.9) y también a CoinRun
(§3), porque HDT hereda `update_actor`.

**Corrección:** `OfflineReplayBuffer(return_to_go=True)` calcula al cargar
cada episodio `rtg = cumsum inversa de reward` (sin descontar, episodio
completo) y `_sample` lo devuelve como séptimo elemento;
`DTAgent.update` lo exige (error explícito si falta) y `update_actor` lo
recibe por `rtg=`. `pretrain_dt.py` y `pretrain_coinrun.py` lo activan.
Verificado con el loader real: `rtg[t] - rtg[t+1] == reward[t]` y el rtg
de entrenamiento ahora va de 0.011 a 3.26 (escalado), el mismo rango que el
target de evaluación (3.6). `test_dt.py`/`test_hdt.py` pasan.

**Otras diferencias con el oficial (sin corregir todavía, no verificadas
como causa):** el oficial recorta el gradiente a 0.25 (`seq_trainer.py`, el
código consultado sí lo confirma); nosotros no. Además, entrenamos solo con
ventanas completas de 20 pasos (el oficial rellena y enmascara), muestreamos
episodios uniformemente (el oficial, proporcional al largo) y nuestra
arquitectura no tiene LayerNorm sobre los embeddings y usa ReLU en la
cabeza de acción.

**Re-entrenamiento lanzado (2026-09-21):** 18 jobs (28426-28443), Adam
simple + normalización de obs, {DT, HDT} × {halfcheetah, hopper, walker2d}
× semillas {1, 2, 3}. Scripts `pretrain_{dt,hdt}_<tarea>_rtgfix_s<seed>.sbatch`,
snapshots en `~/snapshot/<tarea>_medium_expert_{dt,hdt}_rtgfix/<tarea>/<seed>/`.
Se cambia solo el rtg (sin recorte de gradiente) para poder atribuir el
efecto. **Pendiente:** evaluar con `eval_dt.py` y comparar contra §2.9 y el
paper.

**CoinRun también re-entrenado (2026-09-21):** decidido re-entrenar en vez
de solo re-evaluar, porque el bug de rtg afecta el objetivo de
entrenamiento en sí (§2.10), no solo la evaluación — un snapshot entrenado
con rtg roto no se arregla evaluándolo distinto. `pretrain_coinrun.py` ya
pasaba `return_to_go=True` desde el fix, así que alcanzó con relanzar sin
tocar código: jobs 28457 (DT) / 28458 (HDT), mismo protocolo que 27107/
27108 (semilla 1, single-seed, único juego confirmado con el usuario en el
punto 6), snapshots en `~/snapshot/coinrun_{dt,hdt}_rtgfix/`. Pendiente:
evaluar con `eval_coinrun.py` (tasa de éxito por split) una vez terminen y
comparar contra §3 (resultados con el rtg defectuoso, más abajo).

#### Resultados de CoinRun previos a la corrección (jobs 28404/28405)

Con el rtg defectuoso, snapshot 100000, 100 episodios por split, una semilla.
`eval_coinrun.py` reporta "score normalizado" = `(retorno-5)/5` (tabla
`PROCGEN["coinrun"]["easy"] = (5, 10)`, rango **[-1, 1]**, no [0, 1] — como
el reward es binario 0/10, un episodio ganado da score +1 y uno perdido
−1). Como el reward es binario, la tasa de éxito (% de episodios ganados)
es simplemente `retorno_medio / 10` y es más legible que el score
normalizado para este caso puntual; se reportan ambas para no repetir el
error de comparar las dos tandas en escalas distintas (ver más abajo):

| split | DT tasa éxito (score norm.) | HDT tasa éxito (score norm.) |
|---|---|---|
| train | 83% (0.660 ± 0.751) | 91% (0.820 ± 0.572) |
| val   | 76% (0.520 ± 0.854) | 76% (0.520 ± 0.854) |
| test  | 74% (0.480 ± 0.877) | 83% (0.660 ± 0.751) |

Videos en `eval_results/videos_coinrun_{dt,hdt}_100000/` (3 episodios por
split, todos ganados). `eval_coinrun.py` ahora graba video con
`--video-dir`. Curvas de `action_loss` al paso 100000: DT 1.498, HDT 1.279
(`BR` de los logs es `batch_reward`, el reward medio del batch del dataset,
no una métrica del modelo).

#### Resultados de CoinRun tras la corrección (jobs 28685/28686, evaluados 2026-09-22)

Reentrenamiento completado (28685 DT, 28686 HDT, ambos `COMPLETED`, ~2h cada
uno) y evaluado con `eval_dt_coinrun_rtgfix.sbatch`/`eval_hdt_coinrun_rtgfix.sbatch`
(jobs 28839/28840, snapshot 100000, 100 episodios por split, semilla 1,
`--video-dir`). Mismas dos métricas que en la tanda anterior (tasa de
éxito y score normalizado entre −1 y 1), para que la comparación sea
directa:

| split | DT tasa éxito (score norm.) | HDT tasa éxito (score norm.) |
|---|---|---|
| train | 89% (0.780 ± 0.626) | 84% (0.680 ± 0.733) |
| val   | 64% (0.280 ± 0.960) | 83% (0.660 ± 0.751) |
| test  | 70% (0.400 ± 0.917) | 81% (0.620 ± 0.785) |

**Comparación antes → después (tasa de éxito, apples-to-apples):**

| split | DT | HDT |
|---|---|---|
| train | 83% → 89% (+6pp) | 91% → 84% (−7pp) |
| val   | 76% → 64% (**−12pp**) | 76% → 83% (+7pp) |
| test  | 74% → 70% (−4pp) | 83% → 81% (−2pp) |

**Lectura (revisada — la primera versión de esta sección comparaba mal las
dos tandas, ver nota al final):** con el rtg corregido, HDT generaliza
*mejor* que antes (val +7pp) y solo pierde un poco en train; DT es el que
empeora en val/test (−12pp, −4pp) aunque mejora en train (+6pp) — es decir,
parece sobreajustar más a los 200 niveles de entrenamiento en vez de
aprender peor en general. Es consistente con la hipótesis de que la
jerarquía de HDT ayuda a generalizar: con el rtg roto (casi constante, máx.
0.099, §2.10) ambos agentes eran efectivamente behavior cloning puro (sin
señal de retorno real de la que abusar), lo que puede explicar por qué
antes las tres tandas eran más parejas entre sí; con el rtg corregido, DT
sí aprende a condicionarse por retorno pero eso le da más superficie para
memorizar patrones específicos de los niveles vistos, mientras que HDT
aprovecha la señal corregida sin pagar ese costo de generalización.
**Alternativa que no se puede descartar:** CoinRun corre con **una sola
semilla por agente** (§0.1, decisión confirmada con el usuario), así que
nada de esto se puede separar de varianza pura de entrenamiento (init de
pesos, orden de datos) sin repetir con más semillas, como sí se hizo en
D4RL (§2.12).

**Resultados con las 3 semillas (jobs 28845-28848, evaluados 2026-09-22):**
para responder la pregunta de arriba se encolaron 2 semillas más por agente,
mismo protocolo que 28685/28686 (`pretrain_{dt,hdt}_coinrun_rtgfix_s{2,3}.sbatch`,
snapshots en `~/snapshot/coinrun_{dt,hdt}_rtgfix/coinrun/{2,3}/`), y se
evaluaron las 3 juntas con `eval_coinrun_seeds.py` (nuevo, agrega
`eval_coinrun.py` sobre las 3 semillas igual que `eval_seeds.py` hace para
D4RL — job 28881, sin video, 100 episodios por split y semilla):

| agente | split | s1 | s2 | s3 | media ± std (semillas) |
|---|---|---|---|---|---|
| dt | train | 0.780 | 0.800 | 0.740 | **0.773 ± 0.025** |
| dt | val | 0.400 | 0.520 | 0.460 | **0.460 ± 0.049** |
| dt | test | 0.660 | 0.460 | 0.500 | **0.540 ± 0.086** |
| hdt | train | 0.840 | 0.720 | 0.880 | **0.813 ± 0.068** |
| hdt | val | 0.600 | 0.460 | 0.440 | **0.500 ± 0.071** |
| hdt | test | 0.760 | 0.460 | 0.640 | **0.620 ± 0.123** |

**El hallazgo de la corrida de una sola semilla (arriba: "DT sobreajusta,
HDT generaliza mejor") no se sostiene con 3 semillas.** Con más datos, HDT
queda por encima de DT en las 3 splits (train 0.813 vs. 0.773, val 0.500
vs. 0.460, test 0.620 vs. 0.540), no solo en generalización — pero las
diferencias (0.04-0.10) son chicas frente a la desviación entre semillas
(0.025-0.123), así que tampoco alcanza para afirmar con confianza que HDT
sea mejor en general; lo único que cambió de verdad es que **la lectura
de la §2.10 original (basada en una sola semilla) era ruido, no señal**.

**Hallazgo metodológico adicional (afecta a todas las evaluaciones de
CoinRun, no solo esta comparación):** al notar que el valor de "semilla 1"
en esta corrida (dt val 0.400, test 0.660) no coincide con el reportado en
el job original 28839 (dt val 0.280, test 0.400) para el mismo snapshot,
se encontró la causa: `eval_coinrun.py` nunca pasa `rand_seed` a
`procgen.ProcgenEnv(...)`. Sin ese argumento, `procgen/env.py` genera un
seed aleatorio con `random.SystemRandom()` (entropía del SO) en cada
instancia — el `--seed`/`np.random.seed()` del script controla el offset
de episodio pero **no** qué niveles concretos sortea Procgen dentro del
rango `[start_level, start_level+num_levels)`. Es decir: **correr el mismo
snapshot dos veces con el mismo `--seed` da niveles distintos y por lo
tanto números distintos**, una fuente de ruido no documentada hasta ahora,
separada de la varianza entre semillas de entrenamiento. No se corrigió
todavía (requeriría re-evaluar todo lo de CoinRun para que sea comparable) —
queda como mejora pendiente de baja prioridad (pasar `rand_seed=args.seed`
a `ProcgenEnv` en `eval_coinrun.py`/`eval_coinrun_seeds.py` haría los
resultados reproducibles de verdad, a costa de invalidar la comparación
con todo lo ya reportado en este documento).

**Arreglado y re-evaluado (2026-09-25, job 29791, punto 16).**
`eval_coinrun.py::eval_split` ahora pasa `rand_seed=seed` (default 0) a
`ProcgenEnv`. Verificado: dos corridas seguidas del mismo snapshot con el
mismo seed dan exactamente los mismos episodios (pasos y retornos). Todos
los agentes y semillas se evalúan sobre la **misma secuencia de niveles**,
así que la comparación DT vs. HDT pasa a ser pareada. Mismos snapshots,
100 episodios por split y semilla. El log anterior (sin `rand_seed`) se
renombró a `eval_results/eval_coinrun_seeds_rtgfix_sin_randseed.log`.

Score normalizado (tasa de éxito = (score+1)/2, entre paréntesis):

| agente | split | s1 | s2 | s3 | media ± std (semillas) | antes, sin `rand_seed` |
|---|---|---|---|---|---|---|
| dt | train | 0.880 | 0.880 | 0.900 | **0.887 ± 0.009** (94%) | 0.773 ± 0.025 |
| dt | val | 0.380 | 0.580 | 0.600 | **0.520 ± 0.099** (76%) | 0.460 ± 0.049 |
| dt | test | 0.560 | 0.600 | 0.560 | **0.573 ± 0.019** (79%) | 0.540 ± 0.086 |
| hdt | train | 0.880 | 0.900 | 0.840 | **0.873 ± 0.025** (94%) | 0.813 ± 0.068 |
| hdt | val | 0.720 | 0.720 | 0.680 | **0.707 ± 0.019** (85%) | 0.500 ± 0.071 |
| hdt | test | 0.640 | 0.580 | 0.500 | **0.573 ± 0.057** (79%) | 0.620 ± 0.123 |

**Lectura:**
1. **Train y test: empate.** DT y HDT quedan prácticamente iguales (train
   0.887 vs. 0.873; test 0.573 los dos).
2. **Val: HDT mejor.** 0.707 vs. 0.520 (85% vs. 76% de éxito), y las 3
   semillas de HDT (0.68-0.72) quedan por encima de las 3 de DT
   (0.38-0.60). Es la única diferencia consistente, pero no se repite en
   test (niveles ≥250, distribución abierta), así que no alcanza para decir
   que HDT generaliza mejor en general. Val tiene solo 50 niveles y 100
   episodios, y el resultado depende de qué niveles tocaron con
   `rand_seed=0`.
3. **El ruido por muestreo de niveles es del mismo orden que las
   diferencias entre agentes.** Mismos snapshots, solo cambian los niveles:
   DT train pasa de 0.773 a 0.887 y HDT val de 0.500 a 0.707. Con 100
   episodios, el error estándar de la tasa de éxito es ≈5 pp (≈0.1 en
   score). Para una conclusión firme habría que promediar varios
   `rand_seed` de evaluación, o subir los episodios por split.

Log: `eval_results/eval_coinrun_seeds_rtgfix.log` (y
`maskdp_eval_coinrun_seeds_29791.{out,err}`). Sin errores, 38 min en CPU.

**Nota sobre un error de esta sección (corregido 2026-09-22):** la primera
versión reportaba la tanda anterior en tasa de éxito (%) y esta tanda
directamente en "score normalizado" (rango [-1,1], p.ej. "0.28"), sin
avisar que son la misma métrica en dos escalas distintas — eso hacía ver
una caída mucho más grande de la real (p.ej. "76% → 0.28" parece un
desplome, pero 0.28 de score equivale a 64% de tasa de éxito, una caída
real pero bastante menor). Corregido reportando ambas métricas en las dos
tandas.

Videos (3 episodios por split, incluye derrotas: `ep00_ret0.mp4` en DT
test) en `eval_results/videos_coinrun_{dt,hdt}_rtgfix_100000/{train,val,test}/`.
Logs completos en `eval_results/coinrun_{dt,hdt}_rtgfix_100000.log`. Sin
errores en ninguno de los 2 jobs.

### 2.11 Detalle verificado de las otras 4 diferencias con el oficial (mientras se espera §2.9→§9, 2026-09-21)

Mientras los 18 jobs del punto 9 esperan recursos en el cluster, se revisó
con más detalle el código oficial (`kzl/decision-transformer`,
`gym/experiment.py`, `gym/decision_transformer/training/seq_trainer.py`,
`gym/decision_transformer/models/decision_transformer.py`, consultados
2026-09-21) para dejar listo el diagnóstico de costo/riesgo de cada
diferencia del punto 10, antes de decidir cuáles aplicar:

1. **Recorte de gradiente 0.25.** Confirmado en `seq_trainer.py`:
   `torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)` justo
   después de `loss.backward()` y antes de `optimizer.step()`. Costo:
   trivial (una línea en `DTAgent.update`). Riesgo: bajo.
2. **`LayerNorm` sobre los embeddings apilados, antes del transformer.**
   Confirmado en `decision_transformer.py`: `self.embed_ln =
   nn.LayerNorm(hidden_size)` se aplica sobre `stacked_inputs` (R,s,a
   intercalados) antes de entrar al GPT2. Verificado que
   `DecisionTransformer.forward` (`agent/dt.py:191-240`) no tiene nada
   equivalente — arma `x` (intercalado R/s/a) y lo pasa directo a
   `self.blocks`, sin normalizar. Costo: trivial (un `nn.LayerNorm` más su
   aplicación en `forward`). Riesgo: bajo, pero cambia la escala de
   entrada a todas las capas — hay que re-entrenar para medir el efecto,
   no combinar con snapshots viejos.
   (La cabeza de acción **sí** termina en `Tanh` en ambos — lo que decía
   el punto 10 sobre "usa ReLU" es la capa oculta del head de 2 capas
   nuestro, `Linear→LayerNorm→ReLU→Linear→Tanh` vs. el `Linear→Tanh` de
   una sola capa del oficial; no es un bug, es una diferencia de
   capacidad del head. Se deja fuera de la lista de cambios porque no hay
   evidencia de que sea la causa de la brecha.)
3. **Muestreo de episodios proporcional al largo.** Confirmado en
   `experiment.py`: `p_sample = traj_lens[...] / sum(traj_lens[...])` y
   `np.random.choice(..., p=p_sample)` pesa cada trayectoria por su
   duración. Nuestro `OfflineReplayBuffer._sample_episode`
   (`replay_buffer.py:119-124`) usa `random.choice(self._episode_fns)`,
   uniforme por episodio. Costo: moderado (precalcular pesos por largo de
   episodio, pasarlos a `random.choices`/`np.random.choice`). Riesgo:
   bajo, cambio localizado.
4. **Ventanas parciales con padding + máscara de atención.** Confirmado en
   `experiment.py`: `si = random.randint(0, len(traj)-1)` puede caer cerca
   del final, dando una ventana más corta que `K`, que se rellena con
   ceros (padding a la izquierda) y se marca con `attention_mask`;
   `seq_trainer.py` excluye las posiciones enmascaradas de la loss.
   Nuestro `OfflineReplayBuffer._sample` (`replay_buffer.py:129-140`)
   fuerza `idx` a que la ventana completa de `traj_length` entre en el
   episodio (`np.random.randint(0, episode_len - traj_length + 1)`) — no
   hay ventanas parciales ni máscara. Costo: **alto** — requiere enmascarar
   en `_sample`/`make_replay_loader`, en `DecisionTransformer.forward`
   (pasar `attention_mask` a través de `Block`/`CausalSelfAttention`, que
   hoy solo usa la máscara causal fija) y en el cálculo de la loss en
   `update_actor` (excluir posiciones de padding). Riesgo: el más alto de
   los 4 — toca la arquitectura de atención, no solo el pipeline de datos.

**Lectura:** (1) y (2) son cambios de una línea, buenos candidatos a
probar juntos primero (¿la normalización de embeddings o el recorte de
gradiente explican algo de la brecha en hopper/walker2d?). (3) es
razonable si el punto 9 sigue mostrando la brecha. (4) es la más costosa
y la que más se aleja de "un cambio a la vez" — dejar para el final y
solo si (1)-(3) no cierran la brecha, ya que además complica la
comparación DT/HDT (hay que verificar que el masking funcione igual en
la topología jerárquica de HDT). Ninguna de las 4 se implementó todavía
— sigue pendiente de decisión con el usuario (punto 10), ahora con costo
y riesgo estimado por cambio en vez de una lista sin priorizar.

### 2.12 Resultados — re-entrenamiento con rtg corregido (jobs 28429-28442/28708, evaluados 2026-09-22)

Los 18 jobs del punto 9 (§2.10) terminaron `COMPLETED` (walker2d-hdt-seed3
necesitó dos intentos: 28443/28687 se cancelaron por el nodo pineado a
`hydra`, completó como 28708 tras liberar el `nodelist`). Evaluados con
`eval_seeds.py --device cpu` (job 28837, `eval_seeds_rtgfix.sbatch`), que
agrega `eval_dt.py` sobre las 3 semillas y reporta media ± desviación
**entre semillas** del score D4RL normalizado (10 episodios por semilla,
snapshot 100000):

| tarea | agente | s1 | s2 | s3 | media ± std (semillas) | paper (Chen et al. 2021) |
|---|---|---|---|---|---|---|
| halfcheetah | dt | 93.00 | 92.45 | 92.89 | **92.78 ± 0.24** | 86.8 ± 1.3 |
| halfcheetah | hdt | 91.42 | 91.85 | 91.73 | **91.67 ± 0.18** | 86.8 ± 1.3 |
| hopper | dt | 111.68 | 111.36 | 104.76 | **109.27 ± 3.19** | 107.6 ± 1.8 |
| hopper | hdt | 104.24 | 108.85 | 103.45 | **105.51 ± 2.38** | 107.6 ± 1.8 |
| walker2d | dt | 107.75 | 108.06 | 107.65 | **107.82 ± 0.18** | 108.1 ± 0.2 |
| walker2d | hdt | 107.73 | 107.70 | 107.63 | **107.69 ± 0.04** | 108.1 ± 0.2 |

**Lectura:** el fix del rtg (§2.10) cierra por completo la brecha de
hopper/walker2d que motivó la revisión contra el código oficial — las 6
combinaciones quedan dentro o por encima del rango del paper (hopper y
walker2d a menos de 3 puntos, halfcheetah ~5-6 puntos por encima). DT y HDT
quedan muy cerca entre sí en las 3 tareas (diferencia ≤3.8 puntos), sin que
ninguno domine sistemáticamente al otro — no hay evidencia todavía de que
la jerarquía de HDT cueste o ayude en control propioceptivo puro. Ya no
hace falta seguir con el punto 10 (recorte de gradiente, LayerNorm de
embeddings, muestreo proporcional, ventanas con máscara) para cerrar la
brecha con el paper; queda como mejora opcional, no como corrección
necesaria.

Videos del primer episodio de la semilla ganadora por (tarea, agente), con
`best_seed_videos.py --device cpu` (job 28838, `best_seed_videos_rtgfix.sbatch`):
halfcheetah dt→seed1 (92.78), halfcheetah hdt→seed3 (92.23), hopper dt→seed1
(111.59), hopper hdt→seed2 (108.53), walker2d dt→seed2 (108.23), walker2d
hdt→seed1 (107.75) — en `eval_results/videos_best_seed_rtgfix/<tarea>_<agente>_seed<n>.mp4`.
Log completo en `eval_results/eval_seeds_rtgfix.log`/`best_seed_videos_rtgfix.log`.
Sin errores en ninguno de los 4 jobs de evaluación (28837-28840, incluye
también CoinRun — resultados en §2.10).

## 3. Etapa 2 — Régimen visual (decidido: entra en el alcance, ver §3.1)

*(El texto que sigue es el planteo original, antes de la decisión. La
decisión fue incluirla, sobre CoinRun, con la infraestructura de Benjamín:
implementación y resultados en §3.1 y §2.10.)*

Su Etapa II traslada el problema a CoinRun (Procgen) con un codificador
visual IMPALA congelado. Para DT/HDT no existe hoy un dataset offline
visual equivalente cacheado en esta máquina, y D4RL no tiene una variante
visual estándar (existen benchmarks como V-D4RL o el propio Procgen-BC de
Mediratta et al. que ellos usan, pero adoptar uno implica: conseguir el
dataset, adaptar la tokenización de observación a un codificador
convolucional tipo IMPALA, y decidir si la acción sigue siendo continua o
se pasa a un dominio de acción discreta como Procgen). **Esto es una
decisión de alcance que no debería asumirse por defecto** — conviene
decidir con el resto del equipo si esta etapa entra en el trabajo de HDT o
si el aporte se limita a la Etapa 1 + interpretabilidad sobre el régimen
propioceptivo.

**Decisión (2026-09-07): SÍ, la Etapa 2 entra en el alcance del trabajo**,
apoyándonos en la infraestructura que Benjamín ya construyó (no en sus
checkpoints/resultados — la comparación numérica con su trabajo sigue
siendo al final, contra su informe ya terminado, según §0.2. Acá se
reutiliza su *código de conversión de datos y su encoder pretrained*,
que es infraestructura, no resultados).

**Plan concreto para cuando termine la Etapa 1** (investigado
2026-09-07, sin implementar todavía):

1. **Dataset**: Benjamín usa CoinRun (Procgen) desde el dataset offline
   `gen_dgrl` (`coinrun-1M_E`, formato torchrl: observaciones
   `(T+1,3,64,64)` uint8 CHW, acciones discretas `(T,1)` int64 en
   `[0,14]`). Tiene un script de conversión propio,
   `/home/bmancilla/archive/MaskDP/MaskDP_public/convert_coinrun_to_npz.py`,
   que transforma ese formato al **mismo `.npz` que ya consume nuestro
   `replay_buffer.py`** (la clase `OfflineReplayBuffer` es la misma que
   usamos para D4RL — no hay que tocar `replay_buffer.py`). El dataset
   crudo `coinrun-1M_E` no está cacheado en este cluster todavía, hay que
   descargarlo de `gen_dgrl`/torchrl al llegar a esta etapa.
2. **Encoder visual**: copiar `agent/modules/impala_cnn.py` (arquitectura
   IMPALA con `ResidualBlock`/`ConvSequence`) y usar el checkpoint ya
   entrenado `/home/bmancilla/archive/MaskDP/pretrained_encoders/procgen_coinrun_easy_encoder.pt`
   como encoder **congelado** — no hace falta re-entrenarlo.
3. **Acción discreta**: Procgen tiene 15 acciones discretas, pero
   `DecisionTransformer`/`HierarchicalDecisionTransformer` predicen
   acción continua (`action_head` termina en `nn.Tanh()`, entrenado con
   MSE). Hay que agregar una cabeza de clasificación alternativa
   (softmax sobre 15 logits, entrenada con cross-entropy) — Benjamín ya
   resolvió esto en MaskDP (commit "Add discrete action support
   (classification) alongside continuous control"), sirve como
   referencia de diseño aunque el mecanismo de pérdida no es idéntico
   (MaskDP reconstruye, DT/HDT clonan comportamiento).
4. **Dónde entra el encoder en `dt.py`/`hdt.py`**: reemplazando/
   extendiendo `self.state_embed = nn.Linear(obs_dim, n_embd)` — para
   obs de píxeles, la entrada pasaría primero por el IMPALA congelado y
   después por una proyección lineal a `n_embd`, en vez de la proyección
   lineal directa que hay hoy (pensada para obs vectorial de D4RL).

**Avance (2026-09-07, mientras se esperaba que arrancaran los jobs de
Etapa 1):** se decidió avanzar solo en las partes que NO tocan
`agent/dt.py`/`agent/hdt.py` (esos dos archivos los van a leer los jobs
25129-25134 apenas arranquen — si el código cambia antes de que arranquen,
se arriesga a que corran con una versión a medio construir; una vez que
arrancan, el proceso ya cargó el código en memoria y ahí sí es seguro
editar).

- **Dataset descargado**: el `coinrun-1M_E` de gen_dgrl/Facebook Research
  (paper "The Generalization Gap in Offline Reinforcement Learning",
  ICLR 2024) se descarga directo, sin necesitar el paquete `torchrl`
  (que no está instalado en ningún conda env de este cluster), desde
  `https://dl.fbaipublicfiles.com/DGRL/Procgen/Datasets/Compressed/1M/level_200/expert/coinrun.tar.xz`
  (~390MB comprimido, URL confirmada en el código fuente de
  `torchrl.data.datasets.gen_dgrl`). Descargado y extrayendo en
  `raw_data/coinrun/` (`.gitignore`d, igual que `data/`).
- **Script de conversión copiado**: `convert_coinrun_to_npz.py` (raíz del
  repo) — copia del script de Benjamín, mismo contrato de entrada/salida
  documentado en el propio archivo. Único cambio: `--min_length` default
  20 en vez de 64, para que calce con el `traj_length=20` de nuestro
  `dt.yaml`/`hdt.yaml` actual (matcheado al paper de DT en §2.5.1) en vez
  del `traj_length=64` que usa Benjamín en MaskDP.

**Dataset convertido** (2026-09-07):
```
python convert_coinrun_to_npz.py --src_dir raw_data/coinrun/coinrun --out_dir data/coinrun --min_length 20
```
14.466 episodios crudos → **14.348 convertidos y guardados** en
`data/coinrun/` (945MB), 118 descartados por ser más cortos que
`traj_length=20` (0.8% del total, largos descartados entre 3 y 19 pasos).
`raw_data/` (crudo, `.npy`) y `data/coinrun/` (convertido, `.npz`) quedan
`.gitignore`d igual que el resto de `data/`.

**Encoder visual copiado y verificado** (2026-09-07): se copiaron
`agent/modules/impala_cnn.py`, `agent/modules/pixel_encoder.py` y
`agent/modules/load_pretrained_encoder.py` (código de Benjamín, verbatim,
sin modificar) y el checkpoint pretrained a
`pretrained_encoders/procgen_coinrun_easy_encoder.pt`. Smoke test con un
episodio real ya convertido de CoinRun:

```
ImpalaProcgenEncoder(obs_shape=(64,64,3), feature_dim=256) + load_procgen_impala(...)
obs real: (24, 64, 64, 3) uint8  ->  feat: (1, 5, 256)
```

Carga los pesos pretrained (`source: sgoodfriend/ppo-procgen-coinrun-easy`)
sin *surgery* ni mismatch de shapes, y produce features del tamaño
esperado sobre datos reales del dataset ya convertido. Pipeline visual
completo (dataset → encoder pretrained → features) verificado de punta a
punta, sin tocar `dt.py`/`hdt.py`.

**Único paso que falta y que SÍ requiere tocar `dt.py`/`hdt.py`**
(deliberadamente pospuesto hasta que arranquen los jobs 25129-25134, por
la razón explicada arriba): integrar `ImpalaProcgenEncoder` como
reemplazo de `self.state_embed = nn.Linear(obs_dim, n_embd)` cuando la
obs es de píxeles, y agregar la cabeza de acción discreta (softmax sobre
15 logits + cross-entropy, en vez de `nn.Tanh()` + MSE) para CoinRun.

### 3.1 Implementación completa y jobs encolados (2026-09-15)

**Decisión previa de alcance:** el usuario recordó que Benjamín ya tenía
"archivos muy similares" — se investigó su código real (no solo la
mención en prosa de §0.1) en vez de diseñar desde cero: la rama
`upstream/hier-procgen` de este mismo repo (`git fetch upstream
hier-procgen`) tiene su pipeline visual completo (`agent/mdp.py`
extendido con soporte de píxeles+acción discreta, `agent/mdp_bct.py`
— agente de evaluación closed-loop paso a paso, el más análogo a
DT/HDT porque no usa el Algoritmo 1 de enmascaramiento —, `eval_bct.py`).
Se preguntó también si correspondía expandir a los ~9 juegos de Procgen
que el usuario recordaba que Benjamín había probado: revisando
`/home/bmancilla/archive/MaskDP/` (checkpoints, sarfa, encoders
pretrained) todo lo encontrado ahí es CoinRun únicamente — un solo
encoder pretrained, snapshots solo de coinrun, sarfa solo de coinrun.
Sin evidencia en este cluster de otros juegos (podría estar en su
informe/tesis, no accesible acá). **Decisión del usuario: seguir solo con
CoinRun por ahora**, igual que Etapa 1 usó 3 de 7 tareas D4RL por
cobertura — expandir a más juegos queda como paso siguiente si hace
falta, requeriría descargar datasets y conseguir/entrenar un encoder
pretrained por juego adicional.

**Qué se adoptó literal del código de Benjamín** (`agent/mdp.py`/
`mdp_bct.py`/`eval_bct.py`, rama `upstream/hier-procgen`):
- Factory `PixelEncoder(obs_shape, feature_dim, encoder_type=...)` y
  `load_procgen_impala(encoder, ckpt_path, freeze=True)` (ya copiadas en
  `agent/modules/`, §3 más arriba) como reemplazo de `state_embed`/
  `obs_encoder.embed` cuando la obs es de píxeles.
- Acción discreta: `nn.Embedding(num_actions, n_embd)` con squeeze inline
  (`action.long().squeeze(-1)` si trae dim final =1) antes de embeder, en
  vez de una clase envoltorio nueva — mismo estilo que
  `mdp.py::forward_encoder`.
- Cabeza de acción discreta: logits crudos (`nn.Linear` sin `Tanh`),
  softmax implícito dentro de `F.cross_entropy` (no afuera) — mismo
  patrón que `mdp.py::action_head` discreto.
- Splits de evaluación closed-loop y normalización: adoptados completos
  de `eval_bct.py`, no inventados de nuevo — protocolo de Mediratta et
  al. (ICLR 2024): `train` (`start_level=0, num_levels=200`, mismos
  niveles que vio el dataset offline `level_200`), `val`
  (`start_level=200, num_levels=50`, niveles nunca vistos de la misma
  familia), `test` (`start_level=250, num_levels=0`, resto de la
  distribución infinita — generalización real). Diccionario `PROCGEN` de
  rango `(r_min, r_max)` por juego/dificultad para normalizar el retorno
  (`coinrun`/`easy`: `(5, 10)` — confirmado empíricamente sobre el
  dataset offline: reward binario 0/10 por episodio, media 9.55/10).
  `procgen.ProcgenEnv` + wrapper `VecExtractDictObs` (copia literal de
  `baselines.common.vec_env.VecExtractDictObs`) en vez de `gym.make`.

**Qué se mantuvo distinto a propósito:** sin la arquitectura de dos
dimensiones (`enc_n_embd` de encoders vs. `n_embd` de fusión) que tiene
`mdp.py` — nuestro `dt.py`/`hdt.py` ya comparten un solo `n_embd` en toda
la arquitectura; se fijó `n_embd=256` (coincide exacto con la proyección
del checkpoint pretrained) tanto para DT como HDT, sin re-matchear
paridad de parámetros DT/HDT en esta pasada (igual que Etapa 1, que lo
resolvió en dos iteraciones). Sin `encoder_trainable`/fine-tuning (el
encoder queda siempre congelado, no se necesita la generalidad completa
de su optimizador con dos learning rates). Datos: `convert_coinrun_to_npz.py`
sigue agregando la transición dummy al inicio de cada episodio (en vez de
tocar `replay_buffer.py` como hizo Benjamín) — mismo problema, solución
distinta, ya verificada.

**Cambios de código** (`agent/dt.py`, `agent/hdt.py`,
`agent/sequence_encoding.py`): `obs_shape` reemplaza a `obs_dim` (acepta
int o tupla, retrocompatible con los tests existentes de D4RL que pasan
un int); flags `discrete_actions`/`pixel_encoder_type`/
`pretrained_encoder_path`/`label_smoothing` en `transformer_cfg`
(default `False`/`None`/`0.0` vía `getattr`, no rompen `dt.yaml`/
`hdt.yaml` de D4RL); `DTAgent.update_actor`/`act` con rama
cross-entropy/argmax cuando `discrete_actions` (se heredan en `HDTAgent`
sin tocar nada ahí). Verificado con `test_dt.py`/`test_hdt.py`/
`test_sequence_encoding.py` (sin cambios de comportamiento para D4RL) más
`test_dt_coinrun.py` (nuevo, config sintético `(32,32,3)` +
`pretrained_encoder_path=None`, no depende del checkpoint real).

**Cuatro bugs reales encontrados y corregidos durante la implementación**
(todos vía smoke tests, antes de encolar):
1. **`torchvision` faltante en `dt-env`.** `agent/dt.py` ahora importa
   `agent/modules/pixel_encoder.py` incondicionalmente, que importaba
   `torchvision` a nivel de módulo (solo lo usa `ResNetFrozenEncoder`,
   encoder legacy no usado acá) — rompía `eval_dt.py` en `dt-env` (no
   tiene `torchvision` instalado). Fix: import diferido de `torchvision`
   dentro de `ResNetFrozenEncoder.__init__`.
2. **`configure_optimizer` (`agent/dt.py`) no clasificaba `nn.Conv2d`.**
   Solo reconocía `nn.Linear` como "decay"; el encoder IMPALA es todo
   `nn.Conv2d`, así que la aserción final revienta apenas el modelo tiene
   un encoder visual (congelado o no). Fix: (a) parámetros con
   `requires_grad=False` se excluyen por completo de la clasificación
   (nunca reciben gradiente, no necesitan grupo de weight decay); (b)
   `nn.Conv1d/Conv2d/Conv3d` se agregan a `whitelist_modules` (decay)
   para el caso de que el encoder se entrene sin congelar en el futuro.
   Mismo fix que ya tiene `agent/mdp.py::_classify_params` de Benjamín.
3. **Bug de orden de inicialización (encontrado leyendo `mdp.py`, y
   confirmado real en nuestro propio HDT).** `self.apply(self.
   _init_weights)` (llamado desde `initialize_weights()`) reinicializa con
   `xavier_uniform_` CUALQUIER `nn.Linear` del árbol de submódulos,
   incluida la proyección del encoder de píxeles ya cargada con pesos
   pretrained — si el orden de construcción no es cuidadoso, lo pisa. En
   `mdp.py` de Benjamín este bug está presente pero enmascarado
   (`enc_n_embd=128 != 256` fuerza el fallback `ignore_proj=True`, la
   proyección nunca se cargaba en primer lugar). Acá SÍ se manifestó: en
   `HierarchicalDecisionTransformer`, `self.obs_encoder` (que ya carga y
   congela su propio encoder en su propio `__init__`) se construye ANTES
   de que el `self.initialize_weights()` de nivel superior corra
   `self.apply(...)` sobre todo el árbol — confirmado con un test que
   comparaba los pesos cargados contra el checkpoint real (`torch.allclose`
   fallaba antes del fix). **Fix de raíz, no solo de orden**: `_init_weights`
   (en `dt.py`, `hdt.py`, `sequence_encoding.py`) ahora salta cualquier
   módulo cuyo `.weight.requires_grad` ya sea `False` (ya congelado/
   pretrained) — así el invariante no depende de acertar el orden exacto
   de construcción en cada composición de módulos.
4. **`convert_coinrun_to_npz.py` preservaba el nombre crudo de gen_dgrl**
   (ej. `20230329T085223_22588_20_81_10.00.npz`), pero
   `replay_buffer.py::_load()` asume el formato `prefix_idx_len.npz`
   exacto (3 partes separadas por `_`, las dos últimas enteras) — nunca se
   había ejercitado este camino en la verificación previa de §3 (that
   smoke test cargaba un episodio a mano, sin pasar por
   `OfflineReplayBuffer`). Encontrado corriendo `pretrain_coinrun.py` de
   punta a punta. Fix: renombrar a `episode_<idx>_<T>.npz` (misma
   convención que `d4rl_data.py`) y re-correr la conversión completa
   (14.348 episodios, mismo resultado que antes, solo cambia el nombre de
   archivo).

**Archivos nuevos**: `agent/dt_coinrun.yaml`/`agent/hdt_coinrun.yaml`
(`n_embd=256`, `n_head=4`, `n_layer=3`/`n_obs_layer=n_act_layer=n_layer=3`,
`traj_length=20`, `episode_length=1000` — verificado el máximo real sobre
`data/coinrun/`, `return_scale=10` — reward binario 0/10 confirmado
empíricamente, `discrete_actions=true`, `num_actions=15`,
`pretrained_encoder_path` relativo por default, pasado absoluto por CLI
en los sbatch porque hydra cambia el cwd); `pretrain_coinrun.py` +
`pretrain_coinrun.yaml` (copia adaptada de `pretrain_dt.py`, sin
`compute_obs_stats` — no aplica a píxeles); `pretrain_dt_coinrun.sbatch`/
`pretrain_hdt_coinrun.sbatch`; `eval_coinrun.py` (puerto de
`eval_bct.py`/`mdp_bct.py` a `DTAgent`/`HDTAgent`); `test_dt_coinrun.py`.

**Conteo de parámetros** (con la config de arriba, calculado localmente
antes de encolar, útil para discutir en el informe cuánto del modelo es
realmente entrenable vs. heredado del encoder pretrained congelado):

| Modelo | Total | Entrenables | Congelados (encoder IMPALA) | % congelado |
|---|---|---|---|---|
| DT (n_embd=256, n_layer=3) | 3.256.143 | 2.633.999 | 622.144 | 19,1% |
| HDT (n_embd=256, n_obs=n_act=n_top=3) | 8.506.703 | 7.884.559 | 622.144 | 7,3% |

El encoder congelado (622.144 parámetros: convnet IMPALA + proyección a
256) es idéntico en ambos modelos, ya que ninguno lo reentrena — la
diferencia de tamaño total entre DT y HDT (3,26M vs. 8,51M) es enteramente
por la arquitectura del stack transformer (HDT duplica el trabajo con dos
sub-encoders T_obs/T_act antes del stack superior, DT no). **No hay
paridad de parámetros entrenables entre DT y HDT en esta etapa** (a
diferencia de la Etapa 1, §1): HDT entrena ~3x más parámetros que DT
(7,88M vs. 2,63M) — queda documentado como diferencia conocida, no
corregida todavía (ver "Qué se mantuvo distinto a propósito" más arriba,
por qué forzar paridad ahora perdería la carga directa del encoder
pretrained).

**Entorno `procgen-env`**: se intentó `conda create --clone
/home/bmancilla/miniconda3/envs/maskdp_procgen` (env ya verificado
funcionando headless en este cluster) pero falló instalando un paquete
(`scipy`) a mitad de transacción y quedó en rollback; una limpieza
concurrente (`rm -rf` lanzado creyendo que el clonado estaba
simplemente colgado, en vez de leer el log de la tarea en background)
corrompió más el directorio. Se rehizo desde cero, más liviano: env nuevo
(`python=3.8.20`) + `pip install torch==1.13.1+cpu` (CPU-only alcanza,
este env solo hace inferencia para el rollout, el entrenamiento corre en
`maskdp-env` vía SLURM/GPU) + `procgen==0.10.7`/`gym3==0.3.3`/
`numpy==1.23.5` (mismas versiones ya probadas) + `omegaconf` (dependencia
de `utils.py`). Verificado con el mismo smoke test headless
(`procgen.ProcgenEnv(...).reset()`/`.step()`) y con `eval_coinrun.py` de
punta a punta contra snapshots dummy de DT y HDT (2 episodios, 30 pasos,
checkpoint con 2 pasos de gradiente — solo para validar que el pipeline
corre, no resultados).

**Verificación previa a encolar** (mismo criterio que Etapa 1): smoke
test de `pretrain_coinrun.py` en CPU (2 pasos de gradiente, batch_size=2,
subconjunto de 300 episodios para no toparse con el límite de memoria
virtual de la sandbox de esta sesión — el job real en SLURM pide
`--mem=64G`, muy por encima de lo que pesa el dataset completo
descomprimido) para `agent=dt_coinrun` y `agent=hdt_coinrun`, ambos con
el checkpoint pretrained real — corren sin errores, el encoder queda
congelado y con los pesos reales del checkpoint (verificado con
`torch.allclose` contra el archivo, no solo que no lance error). Luego
`eval_coinrun.py` de punta a punta contra esos snapshots dummy, en
`procgen-env`, split `train`, 2 episodios — corre sin errores (retorno
0.00 esperable, checkpoint casi sin entrenar).

**Jobs encolados** (2026-09-15, `--nodelist=hydra`, pendientes por
recursos al encolar):

| Job ID | Corrida | Snapshot dir |
|---|---|---|
| 27107 | DT coinrun | `~/snapshot/coinrun_dt/` |
| 27108 | HDT coinrun | `~/snapshot/coinrun_hdt/` |

100.010 pasos cada uno (`num_grad_steps` default de `pretrain_coinrun.yaml`),
`batch_size=64`, dataset completo (14.348 episodios, `data/coinrun/`).

**Evaluación (2026-09-21):** hecha con `eval_coinrun.py` (env
`procgen-env`, torch solo CPU) sobre los snapshots de 100000, 3 splits ×
2 agentes, 100 episodios por split (jobs 28404/28405). Resultados en §2.10,
marcados como previos a la corrección del return-to-go: los dos
entrenamientos de CoinRun usaron el rtg defectuoso, así que hay que decidir
si se re-entrenan.

## 4. Etapa 3 — Interpretabilidad (AttAttr / SARFA), con adaptaciones

Los dos métodos son trasladables en principio, pero requieren ajustes
porque DT/HDT no tienen la separación encoder/decoder de MaskDP y el
espacio de acción es continuo, no discreto:

- **AttAttr** (atribución sobre pesos de atención, Sección 3.5.2 de su
  trabajo): en MaskDP el punto de atribución es el decodificador
  compartido (idéntico entre topologías). En DT no hay decodificador
  separado — todo el stack de `Block` es la "atribución objetivo". En HDT,
  análogamente, el stack superior (`self.blocks`, después de la fusión)
  es el punto comparable entre topologías, igual que ellos usan el
  decodificador por ser lo único idéntico entre variantes. El objetivo de
  atribución (su logit argmax de acción discreta) no aplica directo: acá
  la cabeza de acción es continua (`nn.Tanh()` sobre una regresión), así
  que el objetivo natural es la acción predicha completa (o su norma L2,
  o una dimensión específica de interés) en vez de un logit de
  clasificación.
- **SARFA** (Sección 3.5.3): está definido sobre una distribución softmax
  de acciones discretas (especificidad/relevancia se calculan sobre esa
  distribución) — **no se traslada directo a control continuo**. Haría
  falta o (a) adaptarlo con una formulación de sensibilidad continua
  (p. ej. cambio en la acción predicha bajo oclusión, medido en L2, en vez
  de cambio en probabilidad), o (b) usar un método de saliencia por
  perturbación distinto pensado para regresión. Esto es trabajo de diseño
  metodológico propio, no una adaptación mecánica.

### 4.1 Diseño de la adaptación a acción continua (2026-09-09, sin implementar todavía)

Ambos métodos necesitan un objetivo escalar (o por dimensión) sobre el
cual atribuir, ya que DT/HDT devuelven un vector de acción continuo vía
`nn.Tanh()` — no hay softmax ni logit argmax del que partir.

**AttAttr:**
- Objetivo de atribución: en vez del logit de la clase argmax, usar la
  acción predicha en la dimensión `j` de interés, `a_pred[j]` (escalar),
  y aplicar la integral de gradientes de Hao et al. 2021 sobre los pesos
  de atención `A_h` del stack objetivo (ya resuelto cuál stack usar, ver
  bullet anterior):
  `Attr_h(A) = A_h ⊙ (1/m) Σ_{k=1}^{m} ∂a_pred[j](k/m · A_h) / ∂A_h`
- **Por dimensión, no agregado desde el inicio**: cada dimensión de
  acción corresponde a una articulación físicamente distinta (un torque
  de un joint). Promediar/agregar antes de tiempo mezclaría atribuciones
  de señales potencialmente independientes o de signo opuesto y taparía
  la pregunta más interesante ("¿el modelo mira el frame correcto del
  pasado para decidir el torque de ESTA articulación?"). Para una cifra
  resumen (p. ej. comparar DT vs. HDT en agregado) se puede normalizar-L2
  las atribuciones por dimensión después, pero el análisis primario es
  por dimensión.
- `m` (pasos de integración): partir de `m=20` (valor típico del paper
  original) y verificar convergencia empírica (la atribución no debería
  cambiar apreciablemente si se duplica `m`); si cambia, subir a
  `m=50-100`.

**SARFA (reformulación completa, no es una adaptación mecánica):**
La intuición central de Puri et al. 2020 — "una región es importante si
perturbarla cambia mucho la salida de interés, y específicamente esa
salida, no todo por igual" — sí se traslada, pero hay que redefinir
specificity/relevance sin distribución softmax:

- **Unidad de perturbación**: en Etapa 1 no hay imagen, así que en vez de
  parches de píxeles se perturban **tokens de la secuencia de entrada**
  (un `state_t`, `action_t` o `return_t` completo dentro de la ventana de
  contexto), reemplazándolo por su valor medio — que es 0, dado que las
  obs ya están normalizadas (hallazgo §2.8). Esto da un mapa de saliencia
  "qué paso pasado importó para la acción de hoy", comparable en espíritu
  al mapa de atención de AttAttr — permite **cruzar ambos métodos como
  validación cruzada** (ver más abajo), igual que en el trabajo de
  referencia.
- Sea `a` la acción predicha sin perturbar, `a'(r)` la acción predicha
  perturbando la región `r`, `Δa(r) = a'(r) - a`, y `j` la dimensión de
  acción de interés:
  - **Specificity_j(r)** = `|Δa_j(r)|` normalizado min-max contra todas
    las regiones candidatas `r'` del mismo estado (mismo criterio de
    normalización que usan métodos de saliencia por oclusión en RL, p.
    ej. Greydanus et al. 2018) → qué tan grande es el efecto de ESTA
    región sobre la dimensión `j`, relativo a las demás regiones.
  - **Relevance_j(r)** = `|Δa_j(r)| / (‖Δa(r)‖₁ + ε)` → qué fracción del
    cambio total de acción se concentra en la dimensión `j` (análogo
    directo a "el cambio es específico de `a*`, no se reparte entre las
    demás acciones" del paper original — ahí eran probabilidades
    discretas, acá es masa de cambio repartida entre dimensiones
    continuas).
  - **SARFA_j(r)** = media armónica(Specificity_j(r), Relevance_j(r)),
    misma fórmula de fusión que el método original.
- Igual que en AttAttr, el análisis primario es por dimensión `j`;
  agregación L2 solo para cifras resumen.

**Validación cruzada de ambos métodos (paso adicional, no estaba en el
plan original):** como estamos rediseñando ambos métodos desde cero (no
es una traslación mecánica de código ya probado), conviene verificar que
AttAttr y SARFA coincidan cualitativamente en qué timesteps pasados
marcan como importantes para el mismo checkpoint/episodio antes de
confiar en los resultados para el análisis DT vs. HDT — si divergen
sistemáticamente, alguno de los dos diseños tiene un problema y hay que
revisarlo antes de seguir.

**Abierto (decidir antes de implementar):** qué dimensión de acción
mostrar cuando se reporte un caso de estudio único en el informe (p. ej.
la de mayor varianza entre episodios, o la que más correlaciona con el
retorno) — no bloquea la implementación del método en general, solo la
elección de qué mostrar como ejemplo.

### 4.2 AttAttr — implementado y validado (2026-09-09)

Implementado en `attattr.py` (nuevo, corre en `maskdp-env`, no depende de
`eval_dt.py`/gym/mujoco_py: toma la ventana de contexto directamente de
un episodio ya convertido en `data/<task>_medium_expert/<task>/`, con la
misma convención de alineación que `OfflineReplayBuffer._sample`). El
parche de `CausalSelfAttention.forward` descrito en §4.1 se hace en
tiempo de ejecución sobre la instancia cargada, sin tocar
`agent/modules/attention.py`.

Validado contra el checkpoint HDT halfcheetah de la primera tanda
(`~/snapshot/halfcheetah_medium_expert_hdt/halfcheetah/1/snapshot_100000.pt`,
job 25130, score 90.44 en §2.6 — elegido por ser un checkpoint ya
conocido como "bueno", sin esperar a que termine la tanda actual de
walker2d):

1. **La reimplementación parcheada es fiel**: corriendo el forward
   parcheado con `alpha=1` (sin intervención real) sobre las 3 capas del
   stack, la salida es idéntica a la del modelo sin parchear
   (`max |diff| = 0.0` en `pred_a`) — descarta que el parche esté
   alterando el cómputo real antes de usarlo para atribución.
2. **Sensibilidad al target**: atribuir `target_dim=0` vs. `target_dim=1`
   da patrones de atribución claramente distintos por capa (confirma que
   el gradiente sí depende de qué dimensión de acción se pide, como debe
   ser — no es una constante que ignora `target_dim`).
3. **Convergencia en `m`**: los valores top de atribución cambian <3%
   entre `m=20` y `m=40` (p. ej. capa 0, t=10/return: 0.002575 vs.
   0.002516) → `m=20` (el default del paper original) ya es suficiente
   acá, no hace falta subir a 50-100.

Pendiente: implementar SARFA (§4.1) y correr la validación cruzada entre
ambos métodos propuesta ahí, antes de usar cualquiera de los dos para
conclusiones DT vs. HDT. La corrida "oficial" para el informe se hace
sobre los checkpoints finales una vez termine Etapa 1 (§2.8) — esto de
acá es solo la validación de que la implementación es correcta.

### 4.3 SARFA — implementado y validado; hallazgo: NO cross-valida con AttAttr (2026-09-15)

Implementado en `sarfa.py` (nuevo, corre en `maskdp-env`, reusa
`OBS_ACTION_DIMS`/`load_agent`/`load_window` de `attattr.py` para operar
sobre exactamente la misma ventana/checkpoint que AttAttr y poder
cruzarlos). Sigue el diseño de §4.1: unidad de perturbación = un token
completo de la secuencia intercalada (`R_t`, `s_t` o `a_t`), reemplazado
por su "valor neutro" — para el estado, el buffer `model.obs_mean` (así
que, tras la normalización z-score interna del modelo, el valor que
efectivamente ve la red es 0, ver §2.8/§4.1); para return-to-go y acción,
0.0 directo (no hay normalización de por medio para esas dos modalidades
en el pipeline). Candidatos = posiciones `0..query_idx` inclusive
(`query_idx = 3(T-1)+1`, el token de estado del último timestep) — el
propio token de acción objetivo (`a_{T-1}`) queda excluido, la máscara
causal ya lo bloquea de influir sobre sí mismo.

**Validación de la implementación** (mismo checkpoint que §4.2,
`halfcheetah_medium_expert_hdt/.../snapshot_100000.pt`, job 25130):
1. **Determinismo**: dos corridas de `sarfa()` sobre la misma
   ventana/target dan resultados idénticos.
2. **Máscara causal**: perturbar el token `a_{T-1}` (fuera del rango de
   candidatos, el que la máscara causal ya bloquea) da `diff=0.0` exacto
   en la predicción — confirma que `query_idx` está bien calculado y que
   la implementación respeta la causalidad del modelo.
3. **Sensibilidad al target**: `target_dim=0` vs. `target_dim=1` dan
   top-5 completamente distintos (mismo criterio que AttAttr en §4.2).
4. **Magnitudes no triviales**: `Δa` finito, con máximo ~1.04 (escala
   razonable para una salida `Tanh`-acotada en `[-1,1]`).

**Validación cruzada contra AttAttr (§4.1, paso pre-registrado antes de
confiar en cualquiera de los dos métodos): el resultado es negativo.**
Corrida sobre 9 ventanas (3 episodios × 3 posiciones de inicio,
`halfcheetah_medium_expert`, mismo checkpoint HDT, `target_dim=0`):

| Comparación | Resultado |
|---|---|
| Correlación de Spearman por posición (AttAttr agregado sobre capas+heads vs. SARFA) | media **-0.075 ± 0.443**, rango [-0.673, 0.668] — sin relación confiable |
| Correlación por capa individual (0/1/2, sin agregar) | media ≈0 ± 0,24–0,39 en las tres — agregar capas no es la causa de la divergencia |
| Acuerdo en qué **modalidad** domina (return/state/action, nivel más grueso) | **3/9 (33%)** — exactamente el nivel de azar para 3 categorías |

Es decir: la divergencia no es un artefacto de cómo se agregan las capas
de AttAttr (se probó agregado y por capa, mismo resultado), ni depende de
la ventana particular elegida (se probó en 9 combinaciones episodio×
posición, con signo de la correlación cambiando de ventana en ventana).
**Los dos métodos, tal como están diseñados hoy, no coinciden de forma
confiable en qué timesteps pasados importan para la predicción de
acción.**

**Patrón observado (pista, no conclusión):** SARFA señala "state" como
modalidad dominante en 7 de 9 ventanas; AttAttr está mucho más disperso
entre return/state/action. Hipótesis más plausible: los tokens de
return-to-go dentro de una ventana de 20 pasos son altamente redundantes
entre sí (`R_t` es una secuencia suavemente decreciente, cada valor es
casi reconstruible a partir de sus vecinos) — el modelo podría aprender a
"atender" mucho a esas posiciones estructuralmente (atención alta →
AttAttr alto), sin que ablacionar una posición individual cambie mucho la
salida porque las posiciones vecinas cargan información casi equivalente
(especificidad baja → SARFA bajo). Esto es consistente con una tensión ya
documentada en la literatura de interpretabilidad de atención en NLP
(Jain & Wallace 2019 "Attention is not Explanation"; Serrano & Smith 2019)
— atribución basada en pesos de atención y saliencia basada en oclusión
miden nociones de "importancia" distintas y no siempre coinciden, incluso
cuando ambas implementaciones son individualmente correctas (como acá,
ver validaciones arriba).

**Consecuencia para el plan (§4.1 ya anticipaba este escenario):** no se
puede usar ninguno de los dos métodos todavía para sacar conclusiones
DT vs. HDT — haría falta, antes de eso, decidir con el usuario alguna de:
1. Aceptar que miden cosas distintas y reportar ambos por separado en el
   informe, con esta limitación documentada explícitamente (no pretender
   que se validan mutuamente).
2. Investigar más a fondo el porqué de la divergencia (p. ej. medir la
   redundancia real entre tokens de return vecinos, o probar SARFA con
   perturbación de *grupos* de tokens en vez de uno a la vez, para ver si
   ablacionar el return-to-go completo — no un solo token — sí tiene
   efecto grande, lo que confirmaría la hipótesis de redundancia).
3. Repetir la validación cruzada sobre DT (no solo HDT) y sobre
   checkpoints de mejor calidad (una vez haya más semillas, §2.9) para
   ver si el patrón es propio de este checkpoint/topología o general.

Ninguna opción está descartada ni elegida — queda pendiente de decisión
antes de usar estos métodos para el análisis de interpretabilidad final.

**En curso (2026-09-15, decisión del usuario tras revisar el hallazgo
anterior):** probar la opción 2 de arriba — la hipótesis de redundancia
entre tokens de return vecinos. Diseño del experimento: en vez de
ablacionar un token de una modalidad a la vez (como hace `sarfa()`),
ablacionar TODOS los tokens de esa modalidad a la vez dentro de la
ventana (`group_ablation()`, nuevo en `sarfa.py`) y comparar el efecto
conjunto (`|Δa_j|` al sacar todo el grupo) contra la suma/máximo de los
efectos individuales ya medidos por `sarfa()`. Si la hipótesis de
redundancia es correcta, "return" debería mostrar un efecto de grupo
desproporcionadamente más grande que la suma de sus efectos individuales
(cada uno solo, casi no importa; todos juntos, sí) — mientras que
"state"/"action" (con menos redundancia interna esperada, cada paso lleva
información más única) no deberían mostrar el mismo patrón tan marcado.
Se corre sobre las mismas 9 ventanas usadas en la validación cruzada de
arriba, para que sea comparable.

**Resultado (`group_ablation()`, mismas 9 ventanas, `target_dim=0`,
promedio):**

| Modalidad | Efecto de grupo `\|Δa_j\|` (media) | Máximo efecto individual (media) | Razón grupo/máx. individual |
|---|---|---|---|
| return | 0,0265 | 0,0040 | **5,37 ± 4,26** |
| action | 0,1113 | 0,0654 | 2,57 ± 1,84 |
| state | 0,8859 | 0,8421 | **1,07 ± 0,61** |

**La hipótesis se confirma parcialmente, con un matiz importante que
cambia la conclusión.** "return" sí es la modalidad más redundante por
lejos (sacar todos los tokens de retorno juntos pesa ~5,4× más que sacar
el más importante de ellos solo — cada uno individual aporta poco porque
sus vecinos cargan información casi equivalente, exactamente el patrón
esperado). "state" no muestra casi nada de este efecto (razón ≈1,07): no
hay redundancia entre los tokens de estado, sacar todos junto pesa casi
lo mismo que sacar solo el más importante — es decir, probablemente hay
un único estado (el más reciente) que ya concentra casi toda la
información relevante.

Pero **el efecto absoluto de "return" (incluso sacándolo TODO junto,
0,0265) sigue siendo ~33× más chico que el efecto de "state" (0,886)**.
Es decir: la redundancia de "return" es real, pero no alcanza para
explicar por qué AttAttr lo marca como la modalidad más importante en
varias ventanas — en términos de cuánto cambia realmente la predicción,
"state" domina con enorme margen incluso después de corregir por
redundancia. **La hipótesis de redundancia explica una parte del
fenómeno (por qué SARFA individual subestima "return") pero no explica
por qué AttAttr lo sobreestima tanto** — ahí sigue habiendo una pregunta
abierta, probablemente relacionada con que la atención puede usar los
tokens de retorno como una especie de "ancla" posicional/estructural sin
que eso se traduzca en un efecto causal real sobre la salida (que es
justamente lo que mide SARFA).

**Conclusión actualizada:** con esta evidencia, **"state" parece ser la
señal genuinamente más importante para la predicción de acción** — tanto
en SARFA individual como en el efecto de grupo, por un margen grande y
consistente. La discrepancia con AttAttr no queda resuelta del todo, pero
ahora hay una pista concreta de por dónde sigue (atención como
estructura vs. atención como causa) en vez de un misterio sin explicar.
Sigue pendiente de decisión con el usuario cómo proceder (ver las 3
opciones más arriba) antes de usar cualquiera de los dos métodos para
una conclusión DT vs. HDT — pero ahora con más elementos para decidir.

### 4.4 Validación cruzada repetida sobre los snapshots con rtg corregido (opción 3 de §4.3, 2026-09-22)

> **⚠ Invalidada (2026-09-25), ver §4.5.** Los scripts de interpretabilidad
> seguían calculando el return-to-go con la fórmula anterior al fix
> (sobre la ventana de 20 pasos y descontada), así que estos snapshots
> recibieron un rtg fuera de distribución. En particular, **"AttAttr marca
> return como dominante en 9/9 ventanas" era un efecto de ese bug** y
> desaparece con el rtg correcto. El texto de abajo queda como registro;
> los logs originales se renombraron a
> `eval_results/interp_crossval_{hdt,dt}_halfcheetah_rtgfix_rtgventana.{log,npz}`.

Punto 11 del resumen de próximos pasos: repetir §4.3 sobre checkpoints
entrenados con el rtg corregido (§2.10/§2.12), para ver si un rtg que sí
distingue episodios buenos de malos (antes casi constante, máx. 0.099)
cambia cuánto pesa "return" para cualquiera de los dos métodos. De paso
cubre la opción 3 que había quedado pendiente en §4.3 (repetir en DT, no
solo HDT, y sobre checkpoints de mejor calidad).

Como la validación cruzada de 9 ventanas se había hecho a mano la primera
vez, se escribió `interp_crossval.py` (nuevo, agrega `attattr.py`/
`sarfa.py` sobre N episodios × M posiciones de inicio y reporta Spearman
agregado, Spearman por capa individual y acuerdo de modalidad dominante)
para poder repetirla de forma reproducible. Corrida sobre
`halfcheetah_medium_expert`, `target_dim=0`, mismas 3 episodios × 3
posiciones que §4.3, sobre los snapshots rtgfix de mejor semilla (§2.12):
HDT semilla 3 (score 92.23) y DT semilla 1 (score 92.78).

| | HDT rtgfix | DT rtgfix | (referencia: HDT rtg roto, §4.3) |
|---|---|---|---|
| Spearman agregado (capas+heads) vs SARFA | −0.067 ± 0.192 | −0.098 ± 0.212 | −0.075 ± 0.443 |
| Acuerdo modalidad dominante | **0/9 (0%)** | **0/9 (0%)** | 3/9 (33%) |
| AttAttr dominante por modalidad | return=9, state=0, action=0 | return=9, state=0, action=0 | disperso (return/state/action) |
| SARFA dominante por modalidad | return=0, state=8, action=1 | return=0, state=7, action=2 | state=7, otros=2 |

**Lectura:** la correlación de Spearman sigue sin mostrar relación
confiable (rango similar, cruza cero) — el rtg corregido **no resuelve**
la divergencia entre los dos métodos, y el patrón es el mismo en DT que
en HDT (descarta que fuera un artefacto de una sola topología o de un
checkpoint particular, opción 3 de §4.3 respondida). Lo que sí cambió,
y de forma consistente en los dos agentes: **AttAttr ahora marca "return"
como la modalidad dominante en el 100% de las ventanas** (antes estaba
disperso entre las 3 modalidades), mientras que **SARFA sigue marcando
"state" como dominante en la gran mayoría** (8/9 y 7/9) — el acuerdo entre
métodos, si algo, empeoró (0/9 vs. 3/9 antes).

Esto es consistente con — y refuerza — la hipótesis de redundancia de
§4.3 (`group_ablation`): con un rtg que ahora sí varía de forma informativa
dentro de la ventana, el modelo aparentemente aprende a **atender** mucho
más a los tokens de retorno (atención alta → AttAttr alto y ahora siempre
dominante), pero sigue sin depender causalmente de ningún token de retorno
individual porque son redundantes entre sí (ablacionar uno solo no cambia
mucho la salida → SARFA bajo). El rtg roto simplemente no le daba al
modelo una razón para atender a esos tokens en absoluto; el rtg corregido
sí, pero el uso que hace de esa atención parece ser más estructural que
causal — la pregunta abierta de §4.3 (por qué AttAttr sobreestima
"return") sigue abierta, ahora con más evidencia de que no depende del rtg
de entrenamiento.

Sin cambios de código en `attattr.py`/`sarfa.py` — solo el script nuevo
`interp_crossval.py` y apuntar a los snapshots rtgfix. Resultados crudos
en `eval_results/interp_crossval_{hdt,dt}_halfcheetah_rtgfix_rtgventana.{log,npz}`
(renombrados, ver aviso al inicio de esta sección).

### 4.5 Interpretabilidad en las 4 tareas con el rtg correcto (2026-09-25)

#### 4.5.1 Bug encontrado: los scripts de interpretabilidad usaban el rtg anterior al fix

`attattr.py`/`sarfa.py` construían el return-to-go de la ventana con
`DTAgent.compute_returns_to_go(reward, discount)`, es decir **sobre los 20
pasos de la ventana y descontado**: justo el rtg defectuoso que se corrigió
en entrenamiento en §2.10. El fix de §2.10 solo tocó `replay_buffer.py` y
`update_actor` (que reciben el rtg del episodio completo desde el buffer).
Los scripts de interpretabilidad lo recalculaban aparte y quedaron
desalineados. Consecuencia: en §4.4 los snapshots rtgfix recibieron un rtg
distinto al de entrenamiento: en D4RL, la suma descontada de 20 pasos
de recompensa (~0.05-0.2 tras dividir por `return_scale=1000`) en vez del
retorno restante del episodio (~1.0 al inicio de un episodio experto).

**Fix:** `attattr.py::load_window` ahora devuelve el rtg crudo del episodio
completo, sin descontar, en `start_idx+i`, idéntico a `episode["rtg"]` de
`OfflineReplayBuffer` con `return_to_go=True`. `att_attr`, `sarfa`,
`group_ablation` y `cross_validate` reciben `rtg` en lugar de
`reward, discount`. Verificado en hopper: rtg de la ventana ≈1018 crudo,
≈1.0 escalado, en distribución.

#### 4.5.2 Extensión a CoinRun (píxeles + acción discreta)

Se agregó soporte para CoinRun a `attattr.py`, `sarfa.py` e
`interp_crossval.py` (`--task coinrun`), siguiendo el código de Benjamín
(ramas `upstream/hier-procgen-attattr` y `upstream/hier-procgen-sarfa`):

| | D4RL (acción continua) | CoinRun (acción discreta) |
|---|---|---|
| Objetivo de atribución | `a_pred[-1, target_dim]` (§4.1) | logit de la acción argmax, como `analysis/attn_attr.py` de Benjamín |
| Fórmula SARFA | reformulación continua de §4.1 | **SARFA original** (Puri et al. 2020) sobre logits, `third_party/sarfa_saliency.py` vendorizado tal cual de su rama |
| Ablación de estado | `obs_mean` (0 tras normalizar) | frame completo difuminado (blur gaussiano σ=3), igual que su saliencia temporal (`eval_sarfa.py`, `temporal_blur_sigma=null → 3.0`) |
| Ablación de acción | 0.0 | índice 4 (NOOP de Procgen; 0 es LEFT+DOWN, no neutro) |
| Ablación de return | 0.0 | 0.0 |

Las ventanas de CoinRun salen del dataset offline de entrenamiento
(`data/coinrun`), igual que en D4RL, no de rollouts en vivo como en
Benjamín. Solo se usan episodios de ≥60 pasos (`--min-len 60`), porque
muchos duran ~20 y no dan lugar a 3 posiciones de inicio distintas.
`interp_crossval.py` ganó `--min-len` y lee el largo del nombre del
archivo (`episode_<idx>_<len>.npz`) para no abrir los ~14k episodios.

No se implementó su saliencia **espacial** (parches difuminados sobre el
frame actual): no es comparable a nivel de token con AttAttr. Queda como
opción para figuras cualitativas.

#### 4.5.3 Protocolo

Snapshots: mejor semilla por (tarea, agente) según §2.12 (halfcheetah dt s1
/ hdt s3, hopper dt s1 / hdt s2, walker2d dt s2 / hdt s1) y semilla 1 en
CoinRun (dt y hdt). `target_dim=0` en D4RL. Dos tamaños de muestra:

- **9 ventanas** (3 episodios × 3 posiciones al 25/50/75%): igual que
  §4.3/§4.4, para comparar directo. Logs:
  `eval_results/interp_crossval_<agente>_<tarea>_rtgfix.{log,npz}`.
- **30 ventanas** (10 episodios × 3 posiciones): la muestra principal; con 9
  los números eran demasiado ruidosos. Logs: `..._rtgfix_n30.{log,npz}`.

Todo en CPU, `maskdp-env`, cada corrida de 30 ventanas tarda unos minutos.

#### 4.5.4 Resultados: validación cruzada AttAttr vs. SARFA (30 ventanas)

| tarea | agente | Spearman agregado (media ± std) | acuerdo modalidad dominante | SARFA dominante (R/S/A) | AttAttr dominante (R/S/A) |
|---|---|---|---|---|---|
| halfcheetah | dt | +0.073 ± 0.295 | 4/30 (13%) | 0/27/3 | 5/3/22 |
| halfcheetah | hdt | +0.033 ± 0.212 | 13/30 (43%) | 2/28/0 | 6/13/11 |
| hopper | dt | +0.108 ± 0.253 | 16/30 (53%) | 6/23/1 | 11/13/6 |
| hopper | hdt | +0.092 ± 0.225 | 12/30 (40%) | 7/23/0 | 5/14/11 |
| walker2d | dt | +0.247 ± 0.211 | 12/30 (40%) | 12/15/3 | 10/5/15 |
| walker2d | hdt | +0.295 ± 0.191 | 9/30 (30%) | 10/14/6 | 4/11/15 |
| coinrun | dt | +0.176 ± 0.214 | 23/30 (77%) | 1/26/3 | 0/25/5 |
| coinrun | hdt | +0.098 ± 0.172 | 9/30 (30%) | 6/18/6 | 2/7/21 |

(R/S/A = return/state/action.)

**Lectura:**
1. **El hallazgo de §4.4 era un artefacto.** Con el rtg correcto, AttAttr ya
   no marca "return" como dominante: en halfcheetah pasa de 9/9 a 5/30
   (DT) y 6/30 (HDT). Sin el bug, AttAttr no tiene ninguna preferencia
   sistemática por return.
2. **La correlación es débil pero ya no es nula, y depende de la tarea.** La
   media es positiva en las 8 combinaciones. En walker2d (~+0.25-0.30) y
   CoinRun DT (+0.18) está claramente por encima de 0 (con 30 ventanas el
   error estándar es ≈0.04). En halfcheetah sigue ≈0. Los dos métodos no
   son intercambiables, pero tampoco miden cosas sin relación.
3. **SARFA marca "state" como dominante en casi todo** (14-28 de 30 según
   tarea). Cuando se ablaciona un token solo, el estado es lo que más mueve
   la acción. AttAttr es más disperso y en varias tareas se inclina por
   "action" (halfcheetah DT, walker2d, CoinRun HDT).
4. **CoinRun DT es el único caso donde los dos métodos coinciden bien**
   (77% de acuerdo, ambos "state"). En CoinRun HDT el acuerdo baja a 30%
   porque AttAttr se va a "action" (21/30).
5. **DT vs. HDT:** no hay un patrón consistente entre tareas. HDT acuerda
   más que DT en halfcheetah y menos en hopper, walker2d y CoinRun. Es una
   sola semilla por agente, así que las diferencias entre agentes de una
   misma tarea no se pueden separar de la varianza entre semillas.

#### 4.5.5 Resultados: ablación por grupos en las 4 tareas (30 ventanas)

Prueba en todas las tareas la hipótesis de redundancia de §4.3: un token
solo pesa poco pero la modalidad entera sí. Nuevo script
`interp_group_ablation.py`, con las mismas 30 ventanas que 4.5.4. Por
modalidad, ablaciona todos sus tokens a la vez ("grupo") y lo compara con
el mayor efecto de un token individual ("max ind"). Efecto: `|Δa_0|` en
D4RL, `|ΔP(a_hat)|` en CoinRun (probabilidad softmax de la acción argmax,
igual que `dP` de SARFA). El logit crudo no sirve en discreto porque los
logits se pueden mover todos juntos sin cambiar la política: en una
ventana de prueba, sacar los return cambió el logit elegido solo 0.09
pero igual cambió la acción. `group_ablation` en `sarfa.py` ganó además
`group_l2` (cambio en toda la salida) y `argmax_changed` (solo discreto).

Medianas sobre 30 ventanas (medias en los logs,
`eval_results/group_ablation_<agente>_<tarea>_rtgfix.{log,npz}`):

| tarea | agente | return: grupo / max ind (ratio) | state: grupo / max ind (ratio) | action: grupo / max ind (ratio) | CoinRun: % ventanas donde cambia la acción (R/S/A) |
|---|---|---|---|---|---|
| halfcheetah | dt | 0.0030 / 0.0016 (2.44) | 0.483 / 0.498 (1.00) | 0.0024 / 0.0017 (1.58) | — |
| halfcheetah | hdt | 0.0062 / 0.0021 (4.11) | 0.153 / 0.144 (0.96) | 0.0055 / 0.0014 (3.87) | — |
| hopper | dt | 0.036 / 0.009 (2.91) | 0.332 / 0.295 (1.09) | 0.018 / 0.005 (3.35) | — |
| hopper | hdt | 0.036 / 0.022 (1.37) | 0.397 / 0.373 (0.81) | 0.016 / 0.009 (2.06) | — |
| walker2d | dt | 0.035 / 0.043 (0.92) | 0.307 / 0.308 (1.00) | 0.012 / 0.006 (2.49) | — |
| walker2d | hdt | 0.039 / 0.027 (1.22) | 0.308 / 0.264 (1.08) | 0.021 / 0.007 (3.36) | — |
| coinrun | dt | 0.042 / 0.008 (5.61) | 0.184 / 0.122 (1.27) | 0.033 / 0.021 (1.57) | 30% / 60% / 23% |
| coinrun | hdt | 0.077 / 0.061 (1.28) | 0.196 / 0.171 (0.99) | 0.041 / 0.025 (1.83) | 33% / 60% / 20% |

**Lectura:**
1. **"State" domina en términos absolutos en todas las tareas**, y su
   ratio grupo/max ≈1: casi todo el efecto lo carga un solo token de
   estado (el más importante individualmente), no una suma de tokens
   redundantes. Consistente con que SARFA marque "state" como dominante
   (4.5.4).
2. **Return y action sí muestran redundancia** (ratio >1 en casi todos los
   casos, hasta ~4-5.6): juntos pesan varias veces más que su mejor token
   individual. La hipótesis de §4.3 se sostiene, pero solo para esas dos
   modalidades.
3. **En D4RL el efecto absoluto de return es chico.** Sacar el return-to-go
   entero mueve `a_0` ~0.003-0.04, frente a ~0.15-0.5 de state (10-100×
   menos). En estas ventanas (datos expertos, en distribución), la acción
   depende casi solo del estado.
4. **En CoinRun el return sí importa:** sacar todo el rtg cambia la acción
   elegida en 30% (DT) y 33% (HDT) de las ventanas, y sacar todas las
   acciones pasadas en ~20%.
5. **Posible diferencia DT vs. HDT en cómo usan el rtg en CoinRun:** en DT
   el efecto del rtg está repartido (max individual 0.008, ratio 5.6); en
   HDT un solo token de return ya pesa bastante (0.061, ratio 1.3). HDT
   también tiene mayor efecto de grupo (0.077 vs. 0.042). Con una sola
   semilla por agente es solo una pista, no una conclusión; habría que
   repetirlo en las 3 semillas.

#### 4.5.6 Estado de la decisión de §4.3 (cómo reportar AttAttr/SARFA)

Las opciones 2 (ablación por grupos) y 3 (DT además de HDT, más checkpoints)
de §4.3 ya están hechas, ahora sobre las 4 tareas y con el rtg correcto.
Resumen para el informe:
- Los dos métodos correlacionan débilmente (Spearman ~0 a +0.3 según
  tarea): no se validan mutuamente, pero tampoco se contradicen.
- SARFA y la ablación por grupos coinciden: el estado es lo que más pesa
  causalmente en todas las tareas. AttAttr (atención) reparte la
  importancia entre las tres modalidades.
- Sigue abierta la opción 1: reportar los dos métodos por separado con esta
  limitación documentada. Es la recomendada; falta la confirmación del
  usuario.


## 5. Selección de variante HDT de referencia

**Cerrado (2026-09-25, decisión del usuario): no aplica.** Solo existe una
variante de HDT y no se va a implementar otra. En particular, no se agrega
fusión por atención cruzada (`CoAttentionBlock` del `agent/mdp.py` de
Benjamín): nuestros dos modelos son Decision Transformers (HDT = flujos
separados por modalidad + auto-atención conjunta), y la comparación es
DT vs. HDT dentro de esa familia, no una réplica de la arquitectura de
MaskDP jerárquico. Que HDT no tenga atención cruzada es parte del diseño,
no una limitación a corregir. El texto original de esta sección queda
abajo como referencia.

Si se terminan implementando varias configuraciones de HDT (paridad vía
`self_attn` únicamente vs. con fusión por atención cruzada, distintos
repartos de dimensión), usar el mismo procedimiento de **agregación por
rangos** de su Sección 3.6.1 para elegir una variante de referencia antes
de pasar a la Etapa 3, en vez de elegir a mano.

## 6. Infraestructura (ya en uso, sin cambios)

PyTorch + Hydra + Weights & Biases, cluster Slurm (`ialab-low`,
`ialab-low-unlimit`), igual que su Sección 3.7. Los datos D4RL ya
convertidos viven en `data/<dataset>/<domain>/episode_*.npz`
(`.gitignore`d, no se suben al repo).

## 7. Etapa BC — behaviour cloning autoregresivo sin return-to-go (desde 2026-10-01)

### 7.0 Origen: reunión del 2026-10-01

Tareas acordadas en la reunión (minuta textual, traducida):

1. Volver a correr **todos los experimentos sin rewards**, para tener un
   setup de behaviour cloning (BC).
2. En CoinRun, **igualar la complejidad en parámetros de la arquitectura
   de Benjamín**.
3. Correr **más semillas en CoinRun** para validar la varianza de la
   evaluación.
4. Pasar a **interpretabilidad** una vez corregido el setup de BC.

Qué cambia respecto de DT/HDT: se elimina el token de return-to-go. Cada
paso aporta 2 tokens `(s_t, a_t)` en vez de 3 `(R_t, s_t, a_t)`, y la
acción se predice desde el token de estado con máscara causal. Siguen
existiendo las dos topologías:

| Topología | Modelo | Equivalente en Benjamín |
|---|---|---|
| unistream | `DecisionTransformer` con `use_rtg: false` (BC-uni) | MaskDP unistream |
| jerárquica | `HierarchicalDecisionTransformer` con `use_rtg: false` (BC-hier) | MaskDP jerárquico (`reduce_enc`) |

Decisiones confirmadas con el usuario (2026-10-01): ambas topologías;
igualar hiperparámetros **y** nº de parámetros de Benjamín.

**Pendiente (el usuario lo está consultando):** si "todos los experimentos"
son D4RL + CoinRun (lo que dice la minuta) o si también entran las tareas
DMC propioceptivas de Benjamín (`/home/bmancilla/archive/MaskDP/maskdp_data/maskdp_train/`).
La parte de código y CoinRun no depende de esa respuesta y avanza mientras
tanto.

### 7.1 Referencias de Benjamín a igualar (verificadas en su código)

> **⚠ Corregido en §7.5 (2026-10-01):** los hiperparámetros de CoinRun de
> abajo son los valores por defecto de `mdp_procgen.yaml`, pero sus
> lanzadores reales (`slurms/procgen/pretrain_{uni,hier}.sh`) usan otros
> (batch 64, contexto 16/64, 800k pasos, dropout 0.1, WD 0.01, early
> stopping). La configuración reportada en su tesis está pendiente de
> confirmar.

- Config CoinRun (`upstream/hier-procgen:agent/mdp_procgen.yaml`):
  `n_embd=256`, `n_head=4`, `traj_length=64`, `batch_size=128`, `lr=1e-4`,
  dropout 0, weight decay 0, PE sinusoidal fija, `num_grad_steps=400010`,
  encoder IMPALA congelado.
- Nº de parámetros entrenables en CoinRun, sin el IMPALA
  (`/home/bmancilla/archive/MaskDP/Hier_procgen/arquitectura_transformer_seccion3.1.md`):
  unistream **4.155.663**, jerárquica `reduce_enc` **4.191.375**.
- Evaluación CoinRun (`upstream/hier-procgen:eval_bct.yaml`): 100 episodios
  por split, train niveles 0-199, val 200-249, test 250+, muestreo con
  temperatura 1.0 (`sample: true`). Nosotros usábamos argmax.
- DMC propioceptivo (solo si entra): `upstream/hierarchical:agent/mdp.yaml`
  (`batch_size=384`, `traj_length=64`, `n_embd=256`, 4 heads), evaluación
  con `Hierarchical/eval_return.py` (retorno normalizado, 10 episodios).

### 7.2 Plan y estimación

Estado al 2026-10-01 entre corchetes.

1. [hecho, §7.3] Código: opción `use_rtg` en `transformer_cfg` de
   `agent/dt.py` y `agent/hdt.py` (default `true`, lo anterior no cambia).
   Se guarda en el `cfg` del snapshot, así que eval e interpretabilidad lo
   recuperan solos. Incluye relleno de episodios cortos y muestreo con
   temperatura en la evaluación.
2. [hecho, §7.3] CoinRun: configs `bc_uni_coinrun`/`bc_hier_coinrun` con los
   hparams de §7.1 y paridad de params (+1,4% / +6,6%).
3. a) D4RL sin R con la misma config que §2.12 (solo se quita R);
   b) DMC de Benjamín, condicional. [esperando la decisión del usuario, §7.0]
4. [hecho, §7.4] Smoke tests para medir tiempos reales: ~32-36 h por
   corrida en 1080 Ti con píxeles.
4b. [hecho, §7.4] Precalcular los embeddings del IMPALA, verificar que
   entrenar sobre ellos equivale a entrenar sobre píxeles y medir el tiempo
   en A40 (~9 h por corrida).
5. Entrenamiento: CoinRun {uni, hier} × 5 semillas = 10 jobs en A40
   (`train_bc_coinrun.sbatch`) [lanzados 2026-10-01, jobs 30561-30570]; D4RL {uni, hier} × 3
   tareas × 3 semillas = 18 jobs [tras la decisión del paso 3].
6. Varianza de evaluación en CoinRun: 5 semillas de entrenamiento × varios
   `rand_seed` de evaluación (`eval_coinrun_seeds.py --eval-seeds`, listo).
7. Interpretabilidad (AttAttr/SARFA, ablación por grupos) con solo
   state/action. [scripts adaptados y verificados, §7.3; falta correrlos
   sobre los modelos entrenados]

Estimación original: ~4-5 días de implementación (+2 si entra DMC),
cómputo de ~1 día para D4RL y ~3-6 días de reloj para CoinRun, ~3-4 días
de eval + interpretabilidad. Total ~2-3 semanas (~3-4 con DMC). La
implementación de CoinRun quedó lista el primer día; la duración de las
corridas quedó en ~9 h cada una en A40; con 2 A40 libres, las 10 corridas
toman ~2 días de reloj.

### 7.3 Implementación (2026-10-01)

**Modelo.** `use_rtg` en `transformer_cfg` de `agent/dt.py` y `agent/hdt.py`
(default `true`, los snapshots anteriores cargan igual). Con `false`:

- no se crean `return_embed` (ni `return_pos_embed` en HDT);
- la secuencia es `(s_1, a_1, s_2, a_2, ...)`, `max_len = 2T`;
- la acción `a_t` se predice desde el token `s_t` (posición `0::2`), con la
  máscara causal de siempre: ve `s_1..s_t` y `a_1..a_{t-1}`;
- `act()`/`update()` ignoran el rtg; como `use_rtg` queda en el `cfg` del
  snapshot, los scripts de evaluación lo recuperan sin flags nuevos.

**Episodios cortos.** En CoinRun la mediana del largo de episodio es 52 y
Benjamín usa contexto 64. Filtrar episodios `< 64` dejaría el 42% de los
episodios (72% de las transiciones), sesgado hacia los largos. Se hace lo
mismo que su `replay_buffer.py` (`upstream/hier-procgen`): opción
`pad_short_episodes` en `replay_buffer.py` (top-level en
`pretrain_*.yaml`, default `false`) que rellena con ceros al final hasta
`traj_length` y agrega una máscara `(T, 1)` al batch; `DTAgent.update_actor`
promedia la loss solo sobre los pasos reales. Con atención causal el
relleno al final no cambia la salida en los pasos reales, así que no hace
falta máscara de atención. Nota: `data/coinrun/` se convirtió con
`--min_length 20` (§3), así que siguen fuera 118 episodios de 3-19 pasos
(0,8%); Benjamín no los filtra. Diferencia menor, se deja así.

**Optimizador.** Benjamín usa AdamW con `weight_decay=0`, betas por
defecto y sin warmup, que es Adam simple → `use_adamw: false`.

**Evaluación.** `eval_coinrun.py --sample --temperature 1.0` muestrea la
acción de `softmax(logits/T)` como su `eval_bct.yaml` (antes solo argmax;
argmax sigue siendo el default). `torch` se siembra por split, así que el
muestreo también es reproducible. `eval_coinrun_seeds.py` ahora acepta
`--agents`, `--seeds`, `--snapshot-pattern` y `--eval-seeds` (varios
`rand_seed` de evaluación, reporta la std entre semillas de entrenamiento y
la std entre `rand_seed`), para la tarea 3 de la reunión. Los defaults
reproducen la evaluación rtgfix anterior.

**Paridad de parámetros en CoinRun** (entrenables, sin el IMPALA congelado,
mismo criterio que la tabla de Benjamín):

| Config | Arquitectura | Params | Referencia Benjamín | Desviación |
|---|---|---|---|---|
| `agent/bc_uni_coinrun.yaml` | DT sin R, `n_embd=256`, 4 heads, 5 bloques | 4.213.007 | unistream 4.155.663 (3 enc + 2 dec a 256) | +1,4% |
| `agent/bc_hier_coinrun.yaml` | HDT sin R, `n_embd=256`, 4 heads, obs 2 + act 2 + conjunto 1 | 4.469.007 | `reduce_enc` 4.191.375 (2 enc por stream + 1 fusión) | +6,6% |

Por qué BC-hier queda en +6,6% y no más cerca: con `n_embd ≠ 256` el
checkpoint del IMPALA no calza (`feature_dim=256`) y `load_procgen_impala`
deja la proyección **aleatoria y congelada** (`ignore_proj=True`). Es la
misma discrepancia que Benjamín anotó para sus variantes jerárquicas
(`Hier_procgen/arquitectura_transformer_seccion3.1.md` §2). Las
combinaciones más cercanas en params la tienen (p. ej. `n_embd=224`, 2+2+2:
4.085.327, -2,5%; `n_embd=208`, 2+2+3: 4.075.775, -2,8%). Se prefiere
mantener el encoder preentrenado completo y la estructura análoga a la
suya. Parte de los params son los embeddings de timestep aprendidos
(`episode_length × n_embd` = 256.000 por stream: 1 en BC-uni, 2 en
BC-hier), que Benjamín no tiene (PE sinusoidal fija).

**Tests.** `test_bc.py` (tensores dummy; DT/HDT, acción continua y discreta):
shapes, ausencia de parámetros de return, rtg ignorado, causalidad (a_t no
influye en su propia predicción, s_t sí), loss enmascarada (cambiar targets
del relleno no cambia la loss), `update` con batch de 6/7 elementos,
`act()` sin rtg, muestreo con temperatura. `test_replay_padding.py` (datos
reales de `data/coinrun`, corrido con `srun`): episodios de 20 y 38 pasos
rellenados y enmascarados correctamente, uno de 70 sin relleno. Los tests
anteriores (`test_dt.py`, `test_hdt.py`, `test_sequence_encoding.py`,
`test_dt_coinrun.py`) siguen pasando.

**Interpretabilidad.** `attattr.py`, `sarfa.py`, `interp_crossval.py` e
`interp_group_ablation.py` asumían 3 tokens por paso (`idx % 3`, query en
`3(T-1)+1`, modalidad "return"). Ahora leen la disposición del modelo con
`attattr.token_modalities(model)` (`(return, state, action)` o
`(state, action)`) y `attattr.query_index`. Las modalidades se identifican
por nombre, también en el `.npz` de `interp_crossval.py` (`dom_attattr`/
`dom_sarfa` pasan de índice a nombre; ningún script lee esos archivos).
Verificado con `srun`: re-correr `interp_crossval.py` sobre DT halfcheetah
rtgfix s1 da el bloque agregado **idéntico** al log guardado
(`eval_results/interp_crossval_dt_halfcheetah_rtgfix.log`), y los 4 scripts
corren sobre snapshots BC (DT y HDT sin entrenar) reportando solo
state/action.

### 7.4 Embeddings precalculados del IMPALA y GPU (2026-10-01)

**Tiempos con píxeles.** Smoke test de 2000 pasos en un 1080 Ti (`yodaxico`,
job 30546) con la config de Benjamín: BC-uni 3,1 pasos/s (400k pasos
≈ 36 h), BC-hier 3,5 pasos/s (≈ 32 h). `action_loss` (cross-entropy sobre
15 acciones; azar = ln 15 ≈ 2,71): BC-uni 2,99 → 2,01, BC-hier 3,55 → 2,01.
El costo lo domina el IMPALA: 128 × 64 = 8.192 frames por paso.

**Decisión (usuario, 2026-10-01):** entrenar en las A40 de `ialab-high`
(`llaima`, QOS `long`) y precalcular los embeddings del IMPALA.

**Por qué precalcular no cambia el entrenamiento.** El IMPALA está
congelado (los 622.144 params con `requires_grad=False`), solo tiene
Conv/MaxPool/ReLU/Linear (nada que cambie entre train y eval), recibe uint8
sin gradiente y no hay aumentación de datos: su salida es una función fija
de cada frame. `precompute_coinrun_embeddings.py` la calcula una vez para
todo `data/coinrun` (FP32 estricto, sin TF32) y escribe `data_emb/coinrun`
(mismos archivos y claves, `observation` pasa de `(T+1, 64, 64, 3)` uint8 a
`(T+1, 256)` float32). `DecisionTransformer.forward` y
`SequenceEncoding.forward` saltan el encoder si la obs de píxeles llega como
`(B, T, 256)`. El snapshot conserva los pesos del IMPALA, así que la
evaluación en Procgen y la interpretabilidad siguen usando píxeles.

Única diferencia: en las posiciones de **relleno** el camino con píxeles
codifica un frame negro (vector ≠ 0) y el de embeddings pone ceros. No
afecta: el relleno va al final, la atención causal impide que lo vean los
pasos reales y la loss lo enmascara.

**Verificación (hecha, 2026-10-01).**

1. `test_precomputed_embeddings.py` (job 30551, A40): embedding guardado vs.
   recalculado, diferencia máxima **0** (idénticos bit a bit) en 3
   episodios; resto de las claves idénticas. BC-uni y BC-hier con sus
   configs reales: logits con píxeles vs. con embeddings difieren como
   máximo **~1e-6**, argmax igual en todos los pasos.
2. Entrenamiento de punta a punta, BC-uni, 300 pasos, semilla 1, TF32
   desactivado, `action_loss` cada 10 pasos (31 puntos):

| Comparación | Máx. \|diff\| de loss | Signo de la diferencia |
|---|---|---|
| píxeles vs. píxeles, misma A40 (job 30560, `e2e_pix_vs_pix_coinrun.sbatch`) | **0** (bit a bit) | — |
| píxeles vs. embeddings, misma A40 (job 30552, `e2e_emb_vs_pix_coinrun.sbatch`) | 4,0e-4 | 22 +, 8 −, 1 = |
| píxeles limpio vs. ruido N(0, 1e-7) en la salida del encoder, semilla 1 (job 30572, `e2e_noise_coinrun.sbatch`) | 4,6e-4 | 16 +, 15 − |
| ídem, semilla de ruido 2 | 4,6e-4 | 16 +, 15 − |

   Loss de referencia: 2,99 → 2,12 en los 300 pasos. Lectura: el
   entrenamiento es determinista en una misma GPU, así que la diferencia
   píxeles vs. embeddings no es ruido entre corridas; viene del ~1e-6 de
   redondeo del embedding (la GPU usa otro algoritmo de convolución para
   8.192 frames por batch que para un episodio suelto). Una perturbación
   del tamaño del último bit de float32 se amplifica al **mismo orden y con
   signo igual de mixto**, y la corrida limpia en una Titan RTX (`hydra`)
   tampoco coincide bit a bit con la de la A40 (2,227556 vs. 2,227549 en el
   paso 50). Precalcular equivale, entonces, a una perturbación de redondeo:
   del mismo tamaño que cambiar de GPU y muy por debajo de la variación
   entre semillas.

**Tiempos en A40 con embeddings** (smoke de 2000 pasos, job 30551): BC-uni
~12 pasos/s, BC-hier ~12-13 pasos/s → ~9 h por corrida de 400k pasos (vs.
~32-36 h en 1080 Ti con píxeles). Las corridas caben en la QOS `regular`
(límite 1 día), con `--time=20:00:00`.

**Entrenamiento.** `train_bc_coinrun.sbatch`, parametrizado por `AGENT` y
`SEED`: 400.010 pasos, snapshots en 0, 5k, 10k, 25k, 50k, 75k, 100k y cada
50k hasta 400k, en `~/snapshot/coinrun_bc_{uni,hier}/coinrun/<seed>/`.

### 7.5 Cambio de enfoque: plug-and-play en el pipeline de Benjamín (2026-10-01)

**Respuesta recibida** (sobre la duda de §7.0): *"use the same tasks as Ben
if that's possible. The dataset and environments should be the same. It
should ideally just be a plug-and-play of a different model."*

**Decisiones del usuario:**
- Nuestro BC-uni/BC-hier se enchufa como un agente más **dentro del código
  de Benjamín**: rama `bc-plugandplay` de este repo, creada desde
  `upstream/hier-procgen` y trabajada en el worktree `~/MaskDP_bc`. Sus
  `pretrain.py`, `replay_buffer.py`, `dmc.py`, `eval_return.py` y
  `eval_bct.py` se usan sin cambios. `main` (este documento, D4RL, nuestro
  pipeline CoinRun) queda intacto.
- Sirve para DMC y CoinRun: su `pretrain.py` entrena ambos.
- Las 10 corridas de CoinRun de §7.4 (jobs 30561-30570) se **cancelaron**
  (s1 y s2 llevaban ~1 h; las demás no habían arrancado). Se rehacen en su
  pipeline.
- La inconsistencia de quadruped (abajo) se replica tal cual y se documenta.

**Verificaciones sobre su pipeline:**
- La rama es idéntica a `/home/bmancilla/archive/MaskDP/Hier_procgen` (0
  líneas distintas en `pretrain.py`, `replay_buffer.py`, `dmc.py`,
  `eval_return.py`, `eval_bct.py`, configs y agentes), o sea, el código de
  sus resultados de CoinRun.
- Sus resultados DMC salen de dos copias más viejas:
  `archive/MaskDP/MaskDP_paper` (unistream) y `archive/MaskDP/Hierarchical`
  (jerárquico). Frente a la rama, `dmc.py` y `eval_return.py` solo difieren
  en valores por defecto (la rama asume píxeles de V-D4RL; se corrige con
  `obs_type=states action_repeat=1`); `pretrain_full.yaml` tiene
  `num_grad_steps=500010` en vez de 400010. El buffer cambió (reparte
  archivos por posición en la lista en vez de por índice del nombre, y
  agrega un corte train/eval/bc opcional), pero simulando ambas lógicas
  cargan **exactamente los mismos episodios** en los 3 dominios con 4 y
  16 workers.
- **Qué datos entran realmente en DMC.** Su `pretrain.py` apunta a
  `maskdp_train/<dominio>` (expert de todas las tareas + semi + sup +
  unsup: 32.600 episodios en cheetah), pero el buffer carga en orden
  alfabético hasta `replay_buffer_size=2.000.000` transiciones, así que en
  la práctica entrena con ~2000 episodios expertos de **una sola tarea** por
  dominio (2004 con 4 workers, su lanzador del paper):

| Dominio | Datos que entran | Tarea de evaluación (`slurms/eval/final/*_return.sh`) |
|---|---|---|
| cheetah | `expert/cheetah_run` | cheetah_run |
| walker | `expert/walker_run` | walker_run |
| quadruped | `expert/quadruped_run` | **quadruped_walk** |

  En quadruped entrena con datos de *run* y evalúa en *walk*; su lanzador
  dice `task=quadruped_walk`, pero `task` solo fija el entorno, no los
  datos. No está claro si es intencional; pendiente de confirmar con su
  tesis o con él.
- **Hiperparámetros DMC** (iguales en sus dos repos): `lr=1e-4`, batch 384,
  contexto 64, `n_embd=256`, 4 heads, sin dropout ni weight decay (Adam o
  AdamW con WD 0), 400k pasos, buffer de 2M con 4 workers. Evaluación: snapshot
  400k, 10 episodios, `action_repeat=1`, `T_cond=12`, `T_pred=12`,
  `replan_freq=1`, 3 semillas por dominio.
- **Hiperparámetros CoinRun: no son los de §7.1.** Sus lanzadores reales:
  `pretrain_uni.sh` (repo `MaskDP_paper`) batch 64, contexto 16, 800k
  pasos, dropout 0.1, WD 0.01, peso de la loss de acción 0.03, early
  stopping por evaluación BCT; `pretrain_hier.sh` (repo `Hier_procgen`)
  batch 64, contexto 64 por defecto (hay snapshots `hier_16` y `hier_64`),
  801k pasos, mismos dropout/WD/early stopping. Las corridas de §7.4 usaban
  los valores por defecto de `mdp_procgen.yaml` (batch 128, contexto 64,
  400k, sin regularización), así que no eran comparables. Falta saber qué
  configuración y qué corridas reporta su tesis.
- **Su evaluación DMC** (`MaskingEvalAgentMultimodal`) tras el calentamiento
  de `T_cond` pasos ejecuta la predicción en la posición `T_cond` de la
  ventana, cuyo estado aún no se observó; podría ser un desfase de un paso.
  Pendiente de revisar. Nuestro agente de evaluación no lo replica: predice
  siempre la acción del estado actual.
- **Entorno conda:** su código usa `np.float` (eliminado en NumPy ≥1.24), así
  que `maskdp-env` no sirve para correrlo; hay que clonar sus entornos
  (`/home/bmancilla/miniconda3/envs/maskdp`, `maskdp_procgen`).

**Implementación en la rama** (`agent/bc_ar.py`, sin tests todavía):
- `BCARModel`: topología `uni` (5 bloques causales sobre `(s, a)`
  intercalados) o `hier` (stack causal por modalidad + bloques conjuntos),
  acción `a_t` desde `s_t`, con los `Block` de su `attention.py` (misma
  cantidad de parámetros que los nuestros; MLP con GELU). Cambios respecto
  a `agent/dt.py`/`agent/hdt.py` de `main`, para no tocar su pipeline:
  posición dentro de la ventana en vez de timestep del episodio (su buffer
  no lo entrega) y sin normalización de observaciones (él tampoco
  normaliza). Acepta embeddings precalculados del IMPALA (§7.4).
- `BCARAgent`: misma interfaz que `MaskedDPMultimodalAgent` (`.model`,
  `.update(replay_iter, step)`, `.train()`), consume su batch
  `(obs, action, reward, discount, next_obs, mask)`, AdamW con WD solo en
  matrices 2D.
- `BCAREvalAgent`: misma interfaz que sus agentes de evaluación (`.mdp`,
  `reset()`, `act(obs)`, y los campos `T_cond`/`T_pred`/`replan_freq` y
  `K`/`temperature`/`sample` que leen `eval_return.py`/`eval_bct.py`);
  ventana deslizante de K pasos, closed-loop.

**Paridad en DMC** (parámetros entrenables; los de Benjamín instanciando sus
modelos con sus configs: unistream `MaskedDP` de `MaskDP_paper`, jerárquico
`reduce_enc` = `MaskedDPMultimodal` de `Hierarchical` con
`enc_n_embd=128`, fusión cruzada):

| Dominio (obs/acción) | BC-uni, 5 bloques | Benjamín unistream | BC-hier, 2+2+1 | Benjamín `reduce_enc` |
|---|---|---|---|---|
| cheetah (17/6) | 3.973.638 | 4.094.487 (−3,0%) | 3.990.022 | 4.161.815 (−4,1%) |
| walker (24/6) | 3.975.430 | 4.098.078 (−3,0%) | 3.991.814 | 4.164.510 (−4,1%) |
| quadruped (78/12) | 3.992.332 | 4.128.858 (−3,3%) | 4.008.716 | 4.187.610 (−4,3%) |

Con la posición dentro de la ventana (64 posiciones en vez de 1000
timesteps) los modelos de CoinRun también quedan en ~−4,5% (la tabla de
§7.3 ya no aplica a la rama).

**Bloqueado hasta tener la tesis:** configuración CoinRun reportada;
variante de referencia en DMC y CoinRun (notricks/jitter/mdrop;
reduce_enc/self/neck); tarea de quadruped; tablas de resultados.

### 7.6 Configuración de Benjamín según su tesis (2026-10-01)

Leída `tesis_benjamin.pdf` (copiada por el usuario a la raíz del repo y
excluida de git). Resuelve lo bloqueado en §7.5.

**Paridad (Tabla 3.1 de la tesis, cheetah):** unistream 4.094.487,
self_attn 4.171.711, reduce_enc 4.161.815, reduce_all 4.032.087. Coinciden
exactamente con lo obtenido instanciando sus modelos (§7.5). **Variante
jerárquica de referencia: reduce_enc** (elegida por agregación de rangos,
su §3.6.1), en DMC y CoinRun.

**DMC propioceptivo** (sus §3.2 y §4.1; Tablas 3.2, 3.3, 3.6, 4.1):
- Tareas: cheetah_run, walker_run, quadruped_walk; solo la partición
  near-expert. La tesis no menciona entrenar quadruped con datos de
  quadruped_run: es un efecto del tope de 2M del buffer (§7.5), no una
  decisión documentada. Se replica tal cual (decisión del usuario) y se
  reporta.
- Entrenamiento: batch 384, contexto 64, `lr=1e-4`, 4 heads, sin
  regularización (dropout y WD en 0), 400k pasos, desde cero.
- Evaluación de retorno: lazo cerrado replanificando en cada paso,
  **100 episodios** por configuración × 3 semillas (sus scripts
  `eval/final/*_return.sh` usaban `num_eval_episodes=10`; se sigue la
  tesis).
- Resultados de referencia, retorno en lazo cerrado (media ± std, 3
  semillas):

| Condición | Arquitectura | cheetah_run | walker_run | quadruped_walk |
|---|---|---|---|---|
| sin aumentación | unistream | 685,67 ± 83,57 | 470,32 ± 126,91 | 922,76 ± 49,31 |
| sin aumentación | reduce_enc | 428,78 ± 372,63 | 452,43 ± 278,72 | 900,78 ± 47,54 |
| jitter + modality dropout | unistream | 639,60 ± 95,38 | 607,52 ± 11,67 | 933,90 ± 5,75 |
| jitter + modality dropout | reduce_enc | 699,01 ± 74,29 | 536,05 ± 29,98 | 922,10 ± 22,47 |

  Para nuestro BC (sin técnicas propias del entrenamiento enmascarado) la
  comparación directa es la condición **sin aumentación**.

**CoinRun** (sus §3.4 y §4.2; Tablas 3.3, 4.6-4.8):
- Datos: expert 1M de Mediratta et al., niveles 0-199, sin
  modificaciones.
- Configuración adoptada: **contexto 16**, batch 64, `lr=1e-4`, dropout
  0,1 (attn/embd/resid), weight decay 0,01 (excluyendo encoder visual,
  sesgos, LayerNorm, embeddings y tokens de máscara), sin aumentación,
  50 épocas con checkpoint cada 5; se reporta el último (época 50 =
  50 × 1M/64 ≈ 781k pasos).
- Encoder visual: IMPALA con convoluciones preentrenadas y **proyección
  lineal aleatoria**, todo congelado, en todas las variantes (control
  experimental: todas reciben la misma representación). En el código de
  su unistream (`MaskDP_paper/agent/mdp.py`) la proyección se carga
  preentrenada (`n_embd=256` coincide con el checkpoint), pero
  `initialize_weights()` corre después y la reinicializa con Xavier (su
  `_init_weights` no salta parámetros congelados). El efecto coincide con
  lo que dice la tesis, aunque en unistream sea por el orden de llamadas.
  Nuestros modelos hacen lo mismo.
- Relleno de episodios cortos con máscara de validez en la loss (igual que
  §7.3).
- Evaluación: lazo cerrado, ventana deslizante, muestreo categórico con
  temperatura 1,0, 100 episodios por split × 3 semillas; se reportan
  train (niveles 0-199) y test (250+); val solo para monitoreo.
- Resultados de referencia (Tabla 4.6, configuración adoptada):

| Arquitectura | Train | Test | Brecha | Exactitud de acción (train) |
|---|---|---|---|---|
| unistream | 9,07 ± 0,12 | 8,07 ± 0,64 | 1,00 | 0,464 |
| reduce_enc | 9,47 ± 0,50 | 8,93 ± 0,31 | 0,54 | 0,369 |
| PPO (referencia externa) | 9,60 | 8,30 | 1,30 | – |

  Tabla 4.7 (sin aumentación): con contexto 64 ambos rinden menos en test
  (reduce_enc 7,40, unistream 7,33 con regularización).

**Configuración que se usará para BC-uni/BC-hier en la rama
`bc-plugandplay`:** la de arriba para cada etapa, con la misma estructura
de 5 bloques (uni) y 2+2+1 (hier) a `n_embd=256`, 4 heads. En DMC, ventana
de evaluación K = 64 (contexto completo, como su evaluación BCT y su
regla de "ventana completa en evaluación"; su script de retorno DMC usaba
`T_cond=12` para su modelo enmascarado).

### 7.7 Rama `bc-plugandplay`: configs, tests, datos y entorno (2026-10-01)

- **Configs** (en la rama): `agent/bc_{uni,hier}.yaml` (DMC) y
  `agent/bc_{uni,hier}_procgen.yaml` (CoinRun) con los valores de §7.6;
  `agent/bc_eval.yaml` para sus `eval_return.py` y `eval_bct.py`. IMPALA con
  convoluciones del checkpoint y proyección aleatoria congelada
  (`impala_projection: random`). Dropout en CoinRun: attn/resid/mlp 0,1 y
  embeddings 0, como su `mdp_procgen.yaml` (la tesis dice "attn/embd/resid";
  se sigue el código).
- **Paridad en CoinRun** (contexto 16, entrenables sin IMPALA): BC-uni
  3.961.103 (−4,7% vs. 4.155.663), BC-hier 3.965.199 (−5,4% vs.
  4.191.375). Algo mayor que en DMC porque nuestro modelo no tiene cabeza de
  reconstrucción de estado; un sexto bloque daría ~+15%, así que se mantienen
  5.
- **Tests** (`test_bc_ar.py`, CPU): causalidad (uni/hier, continua/discreta),
  historial incompleto en evaluación, loss enmascarada, `update` con su
  formato de batch, agente de evaluación (ventana K y acción = modelo sobre
  la ventana), IMPALA (conv = checkpoint, proyección distinta, todo
  congelado), paridad DMC. Todo OK.
- **Datos CoinRun:** su dataset convertido
  (`/home/bmancilla/scratch/procgen_data/coinrun_npz`) ya no existe (solo
  queda un archivo de caché). Se regenera con **su** conversor
  (`archive/MaskDP/MaskDP_public/convert_coinrun_to_npz.py`, copiado tal cual
  a la rama) a partir de nuestra descarga del dataset original
  (`raw_data/coinrun/coinrun`, 14.466 episodios, expert 1M nivel 200 de
  Mediratta et al.). **Supuesto:** `--min_length 1` (no descartar ninguno),
  porque la tesis dice que el relleno permite entrenar con los episodios
  cortos "sin descartarlas"; no se encontró el valor que usó para el
  conjunto de entrenamiento (su lanzador de conversión visible usa 8 para
  validación).
- **Entorno:** su `maskdp_procgen` tiene todo lo necesario (DMC, Procgen,
  torch 1.13 cu113, Hydra); `procgen-env` (nuestro) no sirve (torch CPU, sin
  Hydra). Una primera copia con `srun` se cortó a los 40 min (job 30607,
  sistema de archivos lento); se relanzó como sbatch de 8 h (job 30612,
  `maskdp-ben`). Mientras tanto, el smoke test (job 30613) usa su entorno
  directamente, en solo lectura.
- **Semillas (decisión del usuario):** 3 en DMC y 3 en CoinRun, igual que
  Benjamín → 18 + 6 = 24 corridas.

### 7.8 Hallazgos sobre la evaluación de Benjamín (2026-10-01)

**1. Desfase de un paso en su evaluación de retorno DMC (lazo cerrado).**
`MaskingEvalAgentMultimodal` (`archive/MaskDP/Hierarchical/agent/mdp_return.py`)
y su equivalente unistream (`archive/MaskDP/MaskDP_paper/agent/mdp_return.py`)
usan `deque(maxlen=T_cond)` tanto para los estados como para las acciones.
Con `T_cond=12` (sus `eval/final/*_return.sh`):
- calentamiento (primeros 12 pasos): `n` estados y `n−1` acciones, lee
  `pred_a[n−1]` (acción del estado actual): correcto;
- desde el paso 13: 12 estados `s_{t−11}..s_t` y 12 acciones
  `a_{t−12}..a_{t−1}`; en la grilla intercalada (estados en posiciones
  pares, acciones en impares) cada estado queda emparejado con la acción
  del paso anterior (`(s_t, a_{t−1})`), y se lee `pred_a[T_cond]`, la
  posición de un paso futuro cuyo estado está enmascarado.

No coincide con su Algoritmo 1 (tesis §3.2.3), donde la memoria de acciones va
"siempre un elemento por detrás" y se lee "la acción del instante actual".
Afecta igual a sus cuatro arquitecturas (la comparación interna es justa),
pero sus retornos de lazo cerrado en DMC probablemente están subestimados. Su
evaluación de CoinRun (`BCTEvalAgentMultimodal`, `maxlen=K−1` para acciones,
lee la posición actual) no tiene el problema. Nuestro `BCAREvalAgent` tampoco.
**Decisión del usuario (2026-10-01): no se re-evalúan sus snapshots**; es un
error de su evaluación y se deja documentado, sin cuantificar. Antes de la
decisión se alcanzó a correr una prueba corta (cheetah_run sin aumentación,
semilla 1, **solo 2 episodios**, no concluyente): reduce_enc 758,6 (original)
vs. 770,7 (corregida); unistream 740,1 vs. 817,3. Los scripts
(`reeval_ben/` en el worktree `~/MaskDP_bc`: resolución de sus 72 snapshots
finales y wrapper que corre su `eval_return.py` con la alineación corregida)
quedaron escritos y probados, fuera de git. Al comparar BC (evaluación sin
desfase) contra sus números de DMC hay que mencionar esta diferencia de
protocolo.

**2. Su `eval_bct.py` no fija `rand_seed` en `ProcgenEnv`.** Cada corrida
sortea niveles distintos (mismo problema que el punto 16 en nuestro
`eval_coinrun.py`). Se usa tal cual para ser plug-and-play; es una fuente de
varianza adicional en los números de CoinRun, suyos y nuestros.

**Lanzadores** (rama `bc-plugandplay`, `slurm_bc/`): `train_bc_dmc.sbatch`
y `train_bc_procgen.sbatch` (una semilla por job, configuración de §7.6),
`eval_bc_dmc.sbatch` (su `eval_return.py`, 100 episodios, snapshot 400k) y
`eval_bc_procgen.sbatch` (su `eval_bct.py`, 100 episodios por split, época
50). Snapshots en `~/snapshot/bc_plugandplay/{dmc,procgen}/<agente>/<dominio>/<semilla>/`.

### 7.9 Smoke test y lanzamiento de las 24 corridas (2026-10-01)

**Smoke test** (job 30623, A40, sus scripts sin modificar): tests OK en su
entorno; DMC cheetah_run 2000 pasos + 2 episodios con su `eval_return.py`
(BC-uni 308, BC-hier 367; sin significado con tan poco entrenamiento);
CoinRun 2000 pasos + 2 episodios por split con su `eval_bct.py` (K=16,
muestreo T=1.0). `action_loss` baja (DMC 0,86 → 0,039; CoinRun 3,18 → 2,12).
Un primer intento falló por `use_tb: ${use_tb}` en `agent/bc_eval.yaml`
(la config de `eval_bct.py` no tiene esa clave); corregido.

**Velocidad** (pasos/s; duración proyectada de una corrida):

| Configuración | A40 | Titan RTX |
|---|---|---|
| DMC BC-uni (400k pasos) | 6,6 (~17 h) | 5,8 (~19 h) |
| DMC BC-hier (400k) | 11,3 (~10 h) | 9,9 (~11 h) |
| CoinRun BC-uni (781k) | ~26 (~8,3 h) | 19,5 (~11 h) |
| CoinRun BC-hier (781k) | ~46 (~4,7 h) | 17,1 (~13 h) |

En DMC el cuello de botella es cómputo (batch 384 × 128 tokens), no el
cargador de datos: con 16 workers va igual que con 4. La prueba en 2080 Ti
(job 30649) terminó sin números (salida redirigida a `/dev/null`); no se
depuró porque la Titan RTX alcanza.

**Decisiones del usuario:** repartir en más GPUs (no activar TF32), límite de
24 h por job (sin QOS `long`), 3 semillas en DMC y CoinRun.

**Lanzamiento** (rama `bc-plugandplay`, `slurm_bc/train_bc_{dmc,procgen}.sbatch`,
entorno de Benjamín `/home/bmancilla/miniconda3/envs/maskdp_procgen` en solo
lectura, el mismo para las 24 corridas):

| Grupo | GPU | Jobs |
|---|---|---|
| DMC BC-uni cheetah s1-s2, walker s1-s2 | A40 (`ialab-high`) | 30636, 30637, 30640, 30641 |
| DMC BC-uni cheetah s3 | A40 (`ialab-high-unlimit`, QOS `long`, 24 h) | 30680 |
| DMC BC-uni walker s3, quadruped s1-s3 | Titan RTX (`ialab-low`) | 30655-30658 |
| DMC BC-hier, 3 tareas × 3 semillas | Titan RTX | 30659-30667 |
| CoinRun BC-uni / BC-hier, 3 semillas | Titan RTX | 30668-30673 |

Las semillas de DMC BC-uni quedan repartidas entre A40 y Titan RTX; la
diferencia es de redondeo (§7.4). CoinRun va entero en Titan RTX (sin TF32
en la convolución del IMPALA, a diferencia de la A40).

**Límite de cupos:** la QOS `regular` permite **4 jobs corriendo a la vez por
usuario** (sumando particiones), así que las 24 corridas (~330 h de GPU) van
de a 4: ~3,5 días de reloj. Se agrega un 5.º cupo con la QOS `long` (1 job,
solo en las particiones `-unlimit`, mismos nodos), con `--time=24:00:00` para
respetar la regla de 24 h por job: cheetah s3 BC-uni se movió ahí (job 30639
→ 30680). Estimación con 5 cupos: ~2,8 días.

**Entorno:** la copia propia `maskdp-ben` (job 30612) sigue copiando (~82% a
los 76 min); queda como respaldo si el entorno de Benjamín cambiara durante
las corridas.

## Resumen de próximos pasos concretos

1. [x] Igualar parámetros DT vs HDT (Etapa 0) y documentar la tabla de config.
   Recalculado en §2.5.1 tras matchear hiperparámetros de DT al paper:
   `agent/hdt.yaml` con `n_embd=68`, ~-1.8% de desviación vs DT
   (el `n_embd=88` original de §1.1 quedó obsoleto, calculado contra un
   `dt.yaml` que ya no se usa).
2. [x] Convertir datasets D4RL con `d4rl_data.py`. Se convirtió
   `-expert-v2` para las 3 tareas (§2.1, ya no es el dataset de
   entrenamiento principal pero se conserva) y luego `-medium-expert-v2`
   para las 3 tareas (§2.5.1, dataset final de entrenamiento: 2000/3213/
   2190 episodios halfcheetah/hopper/walker2d).
3. [x] Buscar y citar los números publicados de Decision Transformer sobre
   estos datasets, para la validación de entorno (§2.5). Encontrados
   (Chen et al. 2021, Tabla 2; confirmados contra el código oficial
   `kzl/decision-transformer` y CORL): solo Medium/Medium-Replay/
   Medium-Expert, no "Expert" puro. Resuelto cambiando el dataset de
   entrenamiento a `-medium-expert-v2` (§2.5.1) para tener un número
   citable real.
4. [x] Evaluar con `eval_dt.py` una vez terminen los entrenamientos
   matcheados (jobs 25129-25134, §2.5.1) y comparar contra los números
   publicados (Tabla en §2.5). Resultados y videos en §2.6 (tanda **sin
   AdamW/warmup, sin normalizar obs**) — HDT replica el orden de magnitud
   del paper en las 3 tareas; DT colapsó en halfcheetah. Diagnóstico de
   la curva de pérdida (§2.6) descartó al optimizador como causa y llevó
   a encontrar un problema más grave: **falta normalización de
   observaciones** (§2.8), ausente en todo el pipeline. Los jobs con
   AdamW sin normalizar (25393-25398, §2.7) se cancelaron sin terminar;
   reemplazados por 12 jobs (25406-25417, §2.8) que cruzan
   {Adam simple, AdamW+warmup} × {DT, HDT} × {halfcheetah, hopper,
   walker2d}, todos con la normalización ya activa, para aislar el efecto
   de cada cambio por separado — evaluados en §2.9 (2026-09-15): la
   normalización sola resuelve el colapso de DT en halfcheetah (1.73 →
   89.01), AdamW no aporta de forma consistente encima de la
   normalización, y ninguna combinación alcanza el número del paper en
   hopper/walker2d. Falta repetir con más semillas antes de una
   conclusión firme DT vs. HDT.
5. [x] Extender el entrenamiento/evaluación a hopper y walker2d bajo el
   mismo protocolo (§2.4). Completado con dataset y config matcheados
   (§2.5.1, reemplaza el intento anterior sobre `-expert-v2`, jobs
   25097-25100 cancelados) — resultados en §2.6 (sin AdamW) y §2.7 (con
   AdamW, corriendo).
6. [x] Decidir si la Etapa 2 (régimen visual) entra en el alcance del
   trabajo. Decisión: sí entra, apoyándonos en la infraestructura de
   Benjamín (dataset CoinRun ya convertible, encoder IMPALA pretrained).
   Implementado en §3.1 (2026-09-15): soporte de píxeles+acción discreta
   en `dt.py`/`hdt.py` (basado en el código real de Benjamín, rama
   `upstream/hier-procgen`), jobs 27107 (DT)/27108 (HDT) encolados sobre
   CoinRun (único juego, decisión confirmada con el usuario) —
   evaluación closed-loop hecha (§2.10), pero sobre entrenamientos con el
   return-to-go defectuoso. Re-entrenado con el rtg corregido (§2.10):
   jobs 28685 (DT)/28686 (HDT), evaluados con `eval_coinrun.py` (jobs
   28839/28840, con video) — tasa de éxito DT train/val/test 89%/64%/70%,
   HDT 84%/83%/81% (antes de corregir el rtg: DT 83%/76%/74%, HDT
   91%/76%/83%); HDT generaliza mejor que antes (val +7pp), DT memoriza
   mejor train pero empeora en val/test (§2.10).
7. [x] Diseñar la adaptación de AttAttr/SARFA a acción continua antes de
   empezar la Etapa 3. Diseño completo en §4.1 (2026-09-09): AttAttr con
   objetivo `a_pred[j]` por dimensión de acción; SARFA reformulado por
   completo con perturbación de tokens de la secuencia (no hay imagen en
   Etapa 1) y specificity/relevance redefinidos sin distribución softmax.
   AttAttr implementado y validado en §4.2. SARFA implementado, validado
   individualmente (determinismo, máscara causal, sensibilidad al target)
   y cruzado contra AttAttr en §4.3 (2026-09-15) — **hallazgo: no
   cross-validan** (Spearman por posición ≈0 ± 0,44 sobre 9 ventanas,
   acuerdo de modalidad dominante 3/9 = nivel de azar). No es un bug de
   implementación (ambos pasan sus propias validaciones por separado);
   hipótesis más plausible es que miden nociones de importancia distintas
   (atención vs. oclusión), tensión ya documentada en la literatura de
   interpretabilidad. **Pendiente de decisión con el usuario** antes de
   usar cualquiera de los dos para conclusiones DT vs. HDT (opciones en
   §4.3).
8. [x] Revisar el pipeline contra el código oficial de Decision Transformer
   por la brecha en hopper/walker2d (§2.10, 2026-09-21). Hallazgo: el
   return-to-go de entrenamiento se calculaba sobre la ventana de 20 pasos y
   descontado (0.99), en vez de sobre el episodio completo y sin descontar;
   corregido en `replay_buffer.py`/`agent/dt.py`.
9. [x] Re-entrenamiento con el rtg corregido: 18 jobs (28426-28443/28708),
   Adam simple + norm obs, {DT, HDT} × 3 tareas × semillas {1, 2, 3}.
   Evaluado con `eval_seeds.py`/`best_seed_videos.py` (jobs 28837/28838,
   §2.12): las 6 combinaciones quedan dentro o por encima del rango del
   paper (halfcheetah dt 92.78/hdt 91.67, hopper dt 109.27/hdt 105.51,
   walker2d dt 107.82/hdt 107.69, media ± std entre semillas), cerrando
   la brecha que motivó la revisión del punto 8. DT y HDT quedan muy
   cerca entre sí en las 3 tareas, sin dominancia clara de ninguno.
   Videos de la semilla ganadora por (tarea, agente) en
   `eval_results/videos_best_seed_rtgfix/`. Los 18 jobs (más 28457/28458
   de CoinRun, punto 6) estaban todos pineados a `--nodelist=hydra`, que
   estaba ocupado por jobs de otros usuarios mientras
   `yodaxico`/`ventress`/`scylla` tenían GPUs libres — liberados con
   `scontrol update JobId=<id> ReqNodeList=` (sin reencolar), ya
   corriendo varios en paralelo. El cambio de GPU no afecta los
   resultados (mismo código/semilla, FP32 sin AMP en todo el pipeline;
   la única diferencia real es velocidad, más lenta en los 1080 Ti de
   `yodaxico` que en los Titan RTX de `hydra`) — confirmado ahora que el
   sbatch ya no fija `--nodelist=hydra` como preferencia de infraestructura
   del cluster, no solo para esta corrida.
10. [x] ~~Decidir si se aplican las otras diferencias con el oficial
   (recorte de gradiente 0.25, ventanas cortas con máscara, muestreo
   proporcional al largo) si hopper/walker2d siguen lejos del paper~~ —
   **cerrado sin aplicar (2026-09-25)**: tras el fix del rtg (punto 9,
   §2.12) hopper y walker2d ya quedan dentro del rango del paper, así que
   la condición que motivaba este punto dejó de cumplirse. Las diferencias
   quedan documentadas en §2.11 como mejora opcional, no como corrección
   pendiente.
11. [x] Repetir el análisis de interpretabilidad (§4, AttAttr/SARFA) sobre
   los snapshots con rtg corregido (puntos 9 y 6). Hecho en §4.4
   (2026-09-22), sobre halfcheetah DT y HDT (mejor semilla, §2.12), con
   `interp_crossval.py` (nuevo, agrega las 9 ventanas de forma
   reproducible). **El rtg corregido no resuelve la divergencia** (Spearman
   sigue ≈0, rango similar) y el patrón es igual en DT y HDT — pero sí
   cambió algo: AttAttr ahora marca "return" como modalidad dominante en
   9/9 ventanas en ambos agentes (antes disperso), mientras SARFA sigue
   marcando "state" en la mayoría — el acuerdo entre métodos bajó de 3/9 a
   0/9. Refuerza la hipótesis de redundancia de §4.3: el rtg corregido le
   da al modelo una razón real para *atender* a los tokens de retorno,
   pero no cambia que dependa causalmente de ellos de forma individual.
   **⚠ Invalidado (2026-09-25, punto 13):** los scripts de interpretabilidad
   seguían usando el rtg anterior al fix, y el "return dominante 9/9" de
   AttAttr era un efecto de eso. Resultados válidos en §4.5.
12. [x] Confirmar si la diferencia de generalización DT vs. HDT en CoinRun
   rtgfix (§2.10 — DT sobreajusta más a train, HDT generaliza mejor a
   val/test) es señal real o solo varianza de la única semilla entrenada
   hasta ahora. Jobs 28845-28848 (2 semillas más por agente) + evaluación
   con `eval_coinrun_seeds.py` (nuevo, job 28881) en §2.10 (2026-09-22):
   **el hallazgo de una sola semilla era ruido, no señal** — con 3 semillas
   HDT queda por encima de DT en las 3 splits, pero por márgenes chicos
   frente a la desviación entre semillas. De paso se encontró que
   `eval_coinrun.py` nunca fija `rand_seed` en `ProcgenEnv`, así que cada
   corrida sortea niveles distintos aunque se use el mismo `--seed` —
   fuente de ruido adicional no documentada hasta ahora, pendiente de
   arreglo de baja prioridad (ver detalle en §2.10). **Arreglado en el punto 16.**
13. [x] Corregir el rtg en los scripts de interpretabilidad (§4.5.1,
   2026-09-25). `attattr.py`/`sarfa.py` recalculaban el rtg sobre la
   ventana y descontado (el bug de §2.10), desalineados del entrenamiento.
   Ahora usan el rtg del episodio completo, igual que el buffer. Invalida
   §4.4 (aviso agregado ahí) y el punto 11.
14. [x] Interpretabilidad en hopper, walker2d y CoinRun, y halfcheetah de
   nuevo con el rtg correcto (§4.5, 2026-09-25). Soporte de CoinRun en
   `attattr.py`/`sarfa.py`/`interp_crossval.py`, basado en el código de
   Benjamín (logit argmax, SARFA original de Puri, blur del frame para
   ablacionar estado; §4.5.2). Validación cruzada sobre 9 y 30 ventanas
   (§4.5.4): Spearman débil pero positivo (~0 en halfcheetah, ~+0.25-0.3
   en walker2d), SARFA marca "state" dominante en casi todo, AttAttr más
   disperso. CoinRun DT es el único caso de buen acuerdo (77%).
15. [x] Ablación por grupos en las 4 tareas × DT/HDT
   (`interp_group_ablation.py`, §4.5.5). "State" domina y lo carga un solo
   token (ratio ≈1). Return y action son redundantes (ratio >1), pero en
   D4RL el efecto absoluto del return es 10-100× menor que el de state. En
   CoinRun sacar todo el rtg cambia la acción en ~30% de las ventanas.
   Posible diferencia DT/HDT en cómo usan el rtg en CoinRun, con una sola
   semilla (pendiente, punto 18).
16. [x] Evaluación de CoinRun reproducible: `eval_coinrun.py::eval_split`
   pasa `rand_seed=seed` a `ProcgenEnv` (verificado: dos corridas con el
   mismo seed dan exactamente los mismos episodios). Todos los agentes y
   semillas ven ahora la misma secuencia de niveles. Re-evaluado con
   `eval_coinrun_seeds.py` (job 29791; resultados en §2.10): train y test
   empatados (test 0.573 los dos), HDT mejor solo en val (0.707 vs. 0.520,
   las 3 semillas de HDT por encima de las 3 de DT). El cambio respecto de
   la evaluación anterior (mismos snapshots, otros niveles) es del mismo
   orden que las diferencias entre agentes; para una conclusión firme hay
   que promediar varios `rand_seed` (punto 22).
17. [ ] **Decisión del usuario:** cómo reportar AttAttr/SARFA en el informe
   (§4.5.6). Recomendado: opción 1 de §4.3, reportar los dos por separado
   con la limitación documentada.
18. [ ] Opcional: repetir la interpretabilidad (§4.5) sobre las 3 semillas
   por agente, para separar las diferencias DT vs. HDT de la varianza
   entre semillas, sobre todo el uso del rtg en CoinRun (§4.5.5 punto 5).
19. [ ] Opcional: saliencia espacial en CoinRun (parches difuminados sobre
   el frame, como Benjamín) para figuras cualitativas.
20. [ ] Comparar contra los números reales de Benjamín (§0.1). Bloqueado
   hasta tener su informe o tesis con las tablas.
21. [ ] Informe final consolidado: tablas D4RL y CoinRun, videos,
   interpretabilidad.
22. [ ] Opcional: evaluar CoinRun con varios `rand_seed` de evaluación
   (p. ej. 0-4) o más episodios por split, para separar el ruido por
   muestreo de niveles (≈5 pp de tasa de éxito con 100 episodios) de la
   diferencia DT vs. HDT en val (§2.10).
23. [x] **Etapa BC sin R (§7).** Opción `use_rtg: false` en DT/HDT, relleno
   de episodios cortos, muestreo con temperatura en la eval, con tests (§7.3).
24. [ ] CoinRun BC-uni/BC-hier con hparams y nº de params de Benjamín
   (~4,16M/~4,19M sin IMPALA), 5 semillas por topología (§7.1). Configs y
   embeddings precalculados listos y verificados (§7.3, §7.4); 10 corridas
   lanzadas en A40 (jobs 30561-30570) y **canceladas**: se rehacen dentro
   del pipeline de Benjamín (§7.5), con su configuración real.
25. [x] **Decisión del usuario:** el re-entrenamiento sin R es en D4RL, en el
   DMC de Benjamín o en ambos (§7.0). Respuesta: mismas tareas, datos y
   entornos que Benjamín, como plug-and-play de otro modelo en su pipeline
   (§7.5).
26. [ ] Re-entrenar y evaluar sin R en el benchmark propioceptivo elegido.
27. [ ] Varianza de evaluación en CoinRun: semillas de entrenamiento ×
   `rand_seed` de evaluación (reemplaza el punto 22).
28. [ ] Interpretabilidad sobre los modelos BC (reemplaza los puntos 17-19
   para la etapa nueva).
29. [x] Leer la tesis de Benjamín: configuración CoinRun reportada, variante
   de referencia en DMC y CoinRun, tarea de quadruped y tablas de
   resultados (§7.5). Hecho en §7.6.
30. [x] Rama `bc-plugandplay`: tests de `agent/bc_ar.py`, configs, entorno
   conda de Benjamín, smoke tests en DMC y CoinRun (§7.7, §7.9).
31. [ ] 24 corridas BC en el pipeline de Benjamín (18 DMC + 6 CoinRun, 3
   semillas), lanzadas el 2026-10-01 (§7.9). Al terminar: evaluar con
   `slurm_bc/eval_bc_{dmc,procgen}.sbatch` y comparar contra §7.6.
