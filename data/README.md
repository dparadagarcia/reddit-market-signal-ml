# Datos del proyecto

Esta carpeta separa datos por nivel de procesamiento:

- `raw/`: datos originales descargados (Reddit, mercado).
- `interim/`: transformaciones intermedias.
- `processed/`: dataset semanal final para modelado.
- `external/`: fuentes externas auxiliares (si se usan).

En el repositorio público se mantienen sólo los directorios y sus `.gitkeep`. Los datasets generados localmente no deben subirse a Git salvo que se decida expresamente publicarlos.
