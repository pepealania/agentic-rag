# agentic-rag
# Proof of Concept: LangGraph vs. LlamaIndex Workflows

## Descripción

Este directorio contiene la **prueba de viabilidad (PoC)** realizada para comparar dos alternativas de orquestación de workflows para la arquitectura experimental propuesta:

* **LangGraph**
* **LlamaIndex Workflows**

El objetivo de la PoC es verificar la viabilidad técnica de ambas tecnologías mediante un workflow mínimo de dos nodos:

```text
Consulta
   ↓
Recuperar evidencias
   ↓
Analizar
   ↓
Resultado
```

Además de comprobar la ejecución básica, se evalúan aspectos relacionados con el **estado, trazabilidad, debugging e inspección del flujo de ejecución**, con especial atención a la capacidad de LangGraph para conservar y consultar snapshots del estado mediante checkpoints.

---

## Estructura

```text
PoC/
├── langgraph_poc.ipynb
├── llamaindex_workflows_poc.ipynb
└── README.md
```

### `langgraph_poc.ipynb`

Implementa el workflow utilizando **LangGraph**.

El workflow contiene dos nodos:

1. `recuperar`
2. `analizar`

El estado compartido contiene:

```text
consulta
evidencias
resultado
```

La PoC incorpora un `InMemorySaver` como checkpointer para conservar snapshots de la ejecución.

El historial puede recuperarse mediante:

```python
history = list(
    graph.get_state_history(config)
)
```

Esto permite inspeccionar la evolución del estado durante la ejecución.

---

### `llamaindex_workflows_poc.ipynb`

Implementa el mismo concepto utilizando **LlamaIndex Workflows**.

El workflow contiene dos steps:

1. `recuperar`
2. `analizar`

Los pasos se comunican mediante eventos:

```text
StartEvent
    ↓
EvidenciasEvent
    ↓
StopEvent
```

El workflow utiliza `Context` y `ctx.store` para conservar información de la ejecución y registrar el estado e historial de forma explícita.

---

## Requerimientos

Ambos notebooks están diseñados para ejecutarse en **Google Colab**.

### LangGraph

Versión utilizada:

```text
LangGraph 1.2.9
Python >= 3.10
```

Instalación:

```python
!pip install -q "langgraph==1.2.9"
```

### LlamaIndex Workflows

Versión utilizada:

```text
LlamaIndex Core 0.14.23
Python >= 3.10
```

Instalación:

```python
!pip install -q "llama-index-core==0.14.23"
```

---

## Ejecución

### 1. Abrir los notebooks

Los archivos `.ipynb` pueden abrirse directamente en Google Colab.

### 2. Ejecutar la celda de instalación

Cada notebook contiene una celda inicial para instalar la versión correspondiente del framework.

### 3. Ejecutar el workflow

Ambos notebooks implementan el mismo flujo conceptual:

```text
                 ┌──────────────┐
                 │   Consulta   │
                 └──────┬───────┘
                        ↓
                ┌───────────────┐
                │   Recuperar   │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │    Analizar   │
                └───────┬───────┘
                        ↓
                ┌───────────────┐
                │    Resultado  │
                └───────────────┘
```

No se utiliza un LLM real ni un sistema de recuperación documental en esta primera PoC. Las evidencias son simuladas para evaluar exclusivamente las capacidades de orquestación de cada framework.

---

## Aspectos evaluados

La prueba de viabilidad considera los siguientes aspectos:

| Criterio                        | LangGraph                         | LlamaIndex Workflows       |
| ------------------------------- | --------------------------------- | -------------------------- |
| Instalación                     | ✓                                 | ✓                          |
| Ejecución de workflow           | ✓                                 | ✓                          |
| Flujo de dos nodos              | ✓                                 | ✓                          |
| Gestión de estado               | ✓                                 | ✓                          |
| Comunicación entre pasos        | Estado compartido                 | Eventos                    |
| Debugging básico                | ✓                                 | ✓                          |
| Inspección histórica del estado | **Snapshots / checkpoints**       | Context / store            |
| Trazabilidad                    | **Estado + nodos + transiciones** | Steps + eventos + contexto |
| Control explícito del flujo     | **✓**                             | ✓                          |
| RAG                             | ✓                                 | **✓**                      |

---

## Debugging y trazabilidad

Uno de los objetivos principales de la PoC es evaluar la capacidad de inspeccionar el comportamiento interno del workflow.

### LangGraph

LangGraph utiliza un estado explícito y un mecanismo de checkpoints. Esto permite consultar el estado actual y recuperar el historial de snapshots de una ejecución.

Ejemplo:

```python
history = list(
    graph.get_state_history(config)
)
```

Los snapshots permiten observar cómo evoluciona el estado entre las diferentes etapas del workflow.

Conceptualmente:

```text
Snapshot 0
    ↓
consulta
evidencias = []
resultado = ""

    ↓

Snapshot 1
    ↓
consulta
evidencias = [A, B]
resultado = ""

    ↓

Snapshot 2
    ↓
consulta
evidencias = [A, B]
resultado = "..."
```

Esta capacidad resulta especialmente relevante para la arquitectura experimental porque posteriormente será necesario analizar reintentos, decisiones condicionales, estados intermedios y resultados generados por agentes.

### LlamaIndex Workflows

LlamaIndex permite conservar información utilizando `Context` y `ctx.store`, además de utilizar eventos para comunicar información entre los diferentes steps.

En la PoC se registra explícitamente información como:

```text
consulta
evidencias
resultado
historial
```

Esto permite instrumentar y analizar la ejecución, aunque el historial debe ser gestionado explícitamente mediante el contexto.

---

## Resultado de la PoC

Ambas tecnologías demostraron ser **viables** para implementar el workflow básico de dos nodos.

Sin embargo, se identificó una diferencia importante para los objetivos de esta investigación:

> **LangGraph proporciona una representación más explícita del workflow y mecanismos de persistencia mediante checkpoints que permiten inspeccionar el historial de estados de una ejecución.**

Esta característica facilita el debugging, la trazabilidad y el análisis reproducible del comportamiento del agente.

LlamaIndex Workflows presenta una abstracción adecuada para workflows basados en eventos y ofrece ventajas importantes para aplicaciones centradas en RAG y recuperación documental. No obstante, para la arquitectura propuesta, el control explícito del flujo y la capacidad de inspeccionar estados históricos tienen mayor relevancia.

---

## Decisión tecnológica

A partir de la prueba de viabilidad y de la matriz de decisión ponderada, **LangGraph fue seleccionado como framework de orquestación para la implementación experimental**.

La decisión se fundamenta principalmente en:

1. Control explícito del flujo.
2. Gestión de estado compartido.
3. Soporte para rutas condicionales y ciclos.
4. Capacidad de persistir checkpoints.
5. Inspección del historial de estados.
6. Facilidades para debugging y trazabilidad.
7. Adecuación para experimentar con diferentes estrategias de ejecución.
8. Compatibilidad con la arquitectura de LLM directo, RAG y Agentic RAG.

LlamaIndex Workflows permanece como una alternativa relevante, especialmente para componentes relacionados con recuperación documental y RAG.

---

## Próximos pasos

La PoC de dos nodos constituye únicamente la primera etapa de validación.

La siguiente prueba debería incorporar las características que representan los requisitos reales de la arquitectura:

```text
                    Consulta
                       ↓
                 Recuperación
                       ↓
                    Analista
                       ↓
                ¿JSON válido?
                 /          \
               NO            SÍ
               ↓              ↓
           Reintento         FIN
               ↓
        ¿Límite alcanzado?
           /          \
         NO            SÍ
         ↓              ↓
     Analista         ERROR
```

Esta segunda etapa permitirá evaluar de manera más significativa:

* rutas condicionales;
* reintentos;
* límites de iteración;
* validación de salidas estructuradas;
* persistencia del estado;
* trazabilidad de evidencias;
* debugging de estados intermedios;
* comportamiento de Agentic RAG.

El objetivo final es utilizar el framework seleccionado como capa de orquestación común para comparar experimentalmente:

```text
LLM directo
     vs.
RAG
     vs.
Agentic RAG
```
