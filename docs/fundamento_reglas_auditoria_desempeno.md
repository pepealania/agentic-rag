# Fundamento de las reglas de auditoría del desempeño

## Alcance

Este documento distingue entre (1) el fundamento de la arquitectura RAG y
(2) el fundamento de los criterios de auditoría laboral. Los trabajos de
Agentic RAG citados en el proyecto describen arquitecturas, planificación,
evaluación y autonomía; **no establecen por sí mismos requisitos laborales**.

Las reglas de `src/agentic_rag/deterministic_workflow_prolog_rules.py` son
controles experimentales de calidad de expediente. No son asesoría legal ni
una norma universal de RR. HH. OPM se refiere al servicio federal de EE. UU.;
NIST SP 800-53 es un catálogo de controles de sistemas y organizaciones. Antes
de usarlas en decisiones reales, deben contrastarse con la ley aplicable, el
 convenio colectivo, la política interna y el procedimiento de evaluación.

## Mapeo regla-fuente

| Regla del motor | Sustento pertinente | Qué respalda y qué no respalda |
| --- | --- | --- |
| `evaluador_no_identificado`, `evaluacion_sin_firma`, `version_no_identificada` | ISO 19011:2026; NIST SP 800-53, AU-3 y AU-6 | La evidencia, la documentación y la rendición de cuentas deben permitir conclusiones auditables. Son campos de control propuestos para trazabilidad; esas fuentes no imponen por sí solas una firma específica para toda evaluación laboral. |
| `objetivo_sin_evidencia` | OPM, *Planning* y *Monitoring*; ISO 19011:2026 | OPM indica que los estándares deben ser medibles y verificables, y que el seguimiento compara el desempeño con los estándares establecidos. Solo puede marcarse una brecha si el inventario de objetivos esperado está completo; que un fragmento no aparezca en `top_k` no demuestra que no exista evidencia. |
| `calificacion_discrepante` | ISO 19011:2026; OPM, *Rating* | Una diferencia entre registros es un hallazgo que debe reportarse y contrastarse con criterios y evidencia. La regla no decide qué calificación es correcta ni autoriza a escoger automáticamente una fuente. |
| `evidencia_no_cubre_periodo` | OPM, *Monitoring* y *Rating*; ISO 19011:2026 | OPM describe seguimiento durante el periodo y una calificación basada en el periodo de evaluación. Para concluir una brecha temporal, la fuente de “última evidencia” debe proceder de un repositorio completo y del mismo periodo; una búsqueda RAG parcial no basta. |
| `promedio_inconsistente` | OPM, *Planning* y *Rating* | OPM respalda que los criterios y medidas estén definidos y sean verificables, pero **no prescribe el promedio aritmético ni pesos iguales**. La regla actual calcula una media simple; por tanto, es solo una hipótesis del corpus sintético mientras la organización no declare que los indicadores son comparables, igualmente ponderados y completos. No debe usarse como criterio real de calificación sin esa política explícita. |
| `dato_sensible_no_necesario` | NIST SP 800-53, familia PT (PII Processing and Transparency); marco local de privacidad por determinar | La minimización y el propósito del tratamiento son controles relevantes para información personal. NIST da controles de privacidad para sistemas y organizaciones; no reemplaza la legislación laboral o de protección de datos del país. |
| `acceso_excesivo` | NIST SP 800-53, AC-6 (*Least Privilege*) y familia PT | El acceso debe limitarse al autorizado y necesario según la política aplicable. La lista de personas autorizadas debe venir de una política; el motor no la inventa. |
| `cambio_sin_aprobacion` | NIST SP 800-53, CM-3 (*Configuration Change Control*) y AU-3 | La trazabilidad y aprobación de cambios son controles de gestión de cambios y auditoría de sistemas. Aplicarlos a cambios de evaluación es una decisión de gobernanza organizacional, no una regla laboral universal. |
| `comparacion_global_sin_base` | OPM, *Planning*; ISO 10667-1:2020 | OPM pide estándares medibles, verificables y equitativos; ISO 10667-1 establece requisitos para servicios de evaluación de personas en contextos laborales. Respaldan revisar si existe una base comparable; no prohíben toda comparación entre personas. |

## Condiciones de aplicación

- Cada hecho que ingresa al motor debe tener `employee_id`, `period_id` y
  `document_id`; las salidas incluyen esos documentos como evidencia del
  hallazgo.
- `objetivo_sin_evidencia` requiere el inventario completo de objetivos del
  plan. La evidencia recuperada por similitud no es un inventario exhaustivo.
- `evidencia_no_cubre_periodo` requiere que `latest_evidence` se calcule sobre
  el repositorio completo y con fechas comparables.
- `promedio_inconsistente` solo es interpretable si existe una fórmula
  aprobada, métricas en escala comparable, pesos conocidos y el conjunto
  completo de indicadores. El código actual aún no valida esos supuestos;
  debe tratarse como regla experimental hasta incorporarlos explícitamente.
- Un hallazgo significa “requiere revisión o conciliación”, no “incumplimiento
  probado”, salvo que la política aplicable defina ese efecto y la evidencia
  esté completa.

## Reglas para el corpus sintético

El generador produce 3.000 documentos para 1.000 expedientes. Sus documentos
incluyen `source_metadata.audit_scenario`, una etiqueta asignada por el propio
generador. `evaluate_corpus_documents()` usa esa etiqueta para evaluar la
taxonomía de escenarios y devuelve los IDs de los documentos del expediente
como respaldo. Hay 13 etiquetas y 14 salidas de regla porque el escenario de
minimización de datos produce dos resultados.

| Etiqueta del corpus | Regla semántica | Clasificación |
| --- | --- | --- |
| `trazabilidad_completa` | `trazabilidad_suficiente` | `control_satisfecho` |
| `respaldo_completo` | `objetivos_respaldados` | `control_satisfecho` |
| `evidencia_incompleta` | `cobertura_de_evidencia_insuficiente` | `hallazgo` |
| `objetivos_parcialmente_respaldados` | `cobertura_parcial_de_objetivos` | `hallazgo` |
| `calificaciones_discrepantes` | `conflicto_entre_calificaciones` | `hallazgo` |
| `evidencia_desactualizada` | `cobertura_temporal_insuficiente` | `hallazgo` |
| `metadatos_insuficientes` | `trazabilidad_de_autoria_insuficiente` | `hallazgo` |
| `error_de_calculo` | `calculo_no_reconciliado` | `hallazgo` |
| `minimizacion_datos_personales` | `riesgo_de_minimizacion_de_datos`; `riesgo_de_acceso_excesivo` | `hallazgo` |
| `cambio_sin_trazabilidad` | `cambio_de_evaluacion_sin_trazabilidad` | `hallazgo` |
| `comparacion_no_sustentada` | `comparacion_global_no_sustentada` | `hallazgo` |
| `version_vigente_identificada` | `version_vigente_trazable` | `control_satisfecho` |
| `comparabilidad_limitada` | `limite_de_comparabilidad` | `limitacion_documentada` |

**Límite metodológico:** este modo consume la etiqueta `audit_scenario` como
referencia del escenario. Por ello evalúa reglas contra escenarios conocidos,
pero no demuestra que Prolog haya inferido esos escenarios desde la prosa. Si
`audit_scenario` se usa como entrada de evaluación, no debe tratarse también
como una etiqueta oculta de prueba: eso sería fuga de información. Para medir
inferencia semántica independiente, hay que retirar esa etiqueta de la entrada,
extraer hechos verificables de los documentos, y comparar las conclusiones con
anotaciones de referencia hechas por personas.

## Fundamento arquitectónico

Las fuentes Agentic RAG compartidas respaldan la separación de responsabilidades,
no los criterios laborales anteriores:

- [Singh et al. (2025), *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG*](https://arxiv.org/abs/2501.09136) ofrece una taxonomía de arquitecturas y control/autonomía.
- [Liang et al. (2025), *Reasoning RAG via System 1 or System 2*](https://aclanthology.org/2025.findings-ijcnlp.122/) distingue razonamiento predefinido de interacción agentic dinámica.
- [Nguyen et al. (2025), *MA-RAG*](https://arxiv.org/abs/2505.20096) estudia planificación y colaboración entre agentes especializados.
- [Dong et al. (2025), *RAG-Critic*](https://aclanthology.org/2025.acl-long.179/) estudia crítica de errores y corrección guiada por esos errores.
- [Wang et al. (2025), *Towards Reliable Agents*](https://aclanthology.org/2025.naacl-industry.53/) destaca, entre otros resultados, la disponibilidad de datos en bases de conocimiento acotadas.
- La [documentación de LangGraph sobre workflows y agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) describe workflows como rutas predefinidas y agents como procesos que deciden dinámicamente herramientas y pasos.

Por tanto, este motor de reglas es un evaluador de dominio determinista dentro
de un flujo predefinido. No implementa el planner dinámico de MA-RAG ni la
selección autónoma de acciones correctivas de RAG-Critic.

## Fuentes de auditoría y evaluación

- [ISO 19011:2026, *Guidelines for auditing management systems*](https://www.iso.org/standard/19011). Guía de principios de auditoría, programas y realización de auditorías. La edición 2018 está retirada y fue reemplazada por la edición 2026.
- [ISO 10667-1:2020, *Assessment service delivery — Procedures and methods to assess people in work and organizational settings — Part 1: Requirements for the client*](https://www.iso.org/search.html?q=ISO%2010667-1%3A2020). Requisitos y guía para clientes que usan servicios de evaluación de personas con fines laborales.
- [U.S. Office of Personnel Management, *Performance Management Cycle: Planning*](https://www.opm.gov/policy-data-oversight/performance-management/performance-management-cycle/planning/). Describe estándares medibles, entendibles, verificables, equitativos y alcanzables. Su alcance es el sistema federal estadounidense.
- [U.S. Office of Personnel Management, *Monitoring*](https://www.opm.gov/policy-data-oversight/performance-management/performance-management-cycle/monitoring/) y [*Rating*](https://www.opm.gov/policy-data-oversight/performance-management/performance-management-cycle/rating/). Describen seguimiento contra estándares y evaluación sobre el periodo completo.
- [NIST SP 800-53 Rev. 5, *Security and Privacy Controls for Information Systems and Organizations*](https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final). Catálogo de controles de auditoría, control de acceso, gestión de cambios y privacidad. Es un marco de controles de sistemas/organizaciones, no un estándar de evaluación del desempeño laboral.