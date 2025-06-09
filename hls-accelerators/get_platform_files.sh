#!/bin/bash

DROPBOX_URL="https://www.dropbox.com/scl/fi/z7dhv7vusv8qe5z4nalo1/OperationsVit.zip?rlkey=uhneldas384v1k3wcsgseykwb&st=hcf3ybr4&dl=0"

ARCHIVO="archivo.zip"

echo "Descargando archivo desde Dropbox..."
wget -O "$ARCHIVO" "$DROPBOX_URL"

# Verifica si se descargó correctamente
if [ $? -ne 0 ]; then
  echo "Error al descargar el archivo."
  exit 1
fi


echo "Descomprimiendo $ARCHIVO..."
unzip "$ARCHIVO"

#elimina el archivo comprimido
rm "$ARCHIVO"

echo "Listo."
