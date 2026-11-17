"""Convierte notebooks/tomografia_3d.ipynb en la página docs/notebook_3d.html del sitio.

    python build_notebook_page.py
"""
import re

import nbformat
from nbconvert import HTMLExporter

SRC, OUT = "notebooks/tomografia_3d.ipynb", "docs/notebook_3d.html"
GITHUB = "https://github.com/amoacodc/Tomografia-PINNs/blob/main/notebooks/tomografia_3d.ipynb"

CSS = """<style>
  /* tablas de markdown: ancho según contenido, sin partir fórmulas; scroll horizontal en móvil */
  .jp-RenderedMarkdown table { width:auto !important; max-width:100%; display:block; overflow-x:auto; margin:12px 0; }
  .jp-RenderedMarkdown td, .jp-RenderedMarkdown th { white-space:nowrap !important; padding:6px 14px; text-align:left !important; }
  .site-nav { max-width:1100px; margin:16px auto 0; padding:0 16px; font:15px system-ui,sans-serif; }
</style>"""

nb = nbformat.read(SRC, as_version=4)
for c in nb.cells:
    # el bloque Plotly se sustituye por la versión publicada en el sitio
    if c.cell_type == "code" and "block_faces" in c.source:
        c.outputs = [nbformat.v4.new_output("display_data", data={"text/html":
            '<iframe src="bloque_3d.html" style="width:100%;height:640px;border:0" loading="lazy" '
            'title="Bloque 3D interactivo"></iframe><p style="font-size:.9em">Versión interactiva publicada '
            'en el sitio (malla más fina). <a href="bloque_3d.html">Abrir a pantalla completa</a>.</p>'})]
    if c.cell_type == "markdown" and "> GitHub no muestra" in c.source:
        c.source = c.source.split("> GitHub no muestra")[0].rstrip()

html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb)
html = re.sub(r"<title>.*?</title>", "<title>Notebook tomografía 3D</title>", html, count=1)
html = html.replace("</head>", CSS + "\n</head>", 1)
nav = (f'<div class="site-nav"><a href="index.html">← Volver al sitio</a> · '
       f'<a href="{GITHUB}">Ver el .ipynb en GitHub</a></div>')
html = re.sub(r"(<body[^>]*>)", r"\1" + nav, html, count=1)
open(OUT, "w").write(html)
print(f"Guardado {OUT} ({len(html) // 1024} KB)")
