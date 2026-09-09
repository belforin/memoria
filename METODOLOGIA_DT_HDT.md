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

## 3. Etapa 2 — Régimen visual (abierto, pendiente de decisión)

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

## 5. Selección de variante HDT de referencia

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
   de cada cambio por separado — evaluación pendiente.
5. [x] Extender el entrenamiento/evaluación a hopper y walker2d bajo el
   mismo protocolo (§2.4). Completado con dataset y config matcheados
   (§2.5.1, reemplaza el intento anterior sobre `-expert-v2`, jobs
   25097-25100 cancelados) — resultados en §2.6 (sin AdamW) y §2.7 (con
   AdamW, corriendo).
6. [x] Decidir si la Etapa 2 (régimen visual) entra en el alcance del
   trabajo. Decisión: sí entra, apoyándonos en la infraestructura de
   Benjamín (dataset CoinRun ya convertible, encoder IMPALA pretrained).
   Plan concreto en §3, a implementar después de terminar Etapa 1.
7. [ ] Diseñar la adaptación de AttAttr/SARFA a acción continua antes de
   empezar la Etapa 3.
