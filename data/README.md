# Datos

`dataset.xlsx` contiene 19 908 préstamos de Lending Club con la variable objetivo `loan_status`
(`Fully Paid` = 0, `Charged Off` = 1). Se usa tal como fue entregado en el curso; el notebook
excluye 3 filas con etiqueta vacía o no reconocida.

Proporción de impagos: 14.8 %.

El notebook busca el archivo como `dataset.xlsx` en su mismo directorio o en las rutas de Google Drive
indicadas en la celda de carga. Para ejecutarlo localmente desde `notebooks/`, copia o enlaza el archivo:

```bash
cp data/dataset.xlsx notebooks/
```
