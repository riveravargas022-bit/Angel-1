# Olympikus Lumi (lila / menta) — gemelo digital 3D

Modelo 3D fotorrealista de la zapatilla **Olympikus Lumi** (referencia `652LUMILILA`), reconstruido a partir de las
5 fotos de estudio entregadas (lateral, medial, 3/4, cenital y suela) y renderizado con **Blender 5.0 / Cycles**.

## Contenido

| Carpeta / archivo | Qué es |
|---|---|
| `renders/` | 10 renders de producto a 1800 px (PNG sobre blanco con sombra de contacto) |
| `renders/transparente/` | Las mismas vistas en PNG con transparencia, para e-commerce o maquetación |
| `comparacion/` | Hoja *referencia vs. render* con cámaras calibradas a cada foto y métricas de silueta |
| `modelo/olympikus_lumi.blend` | Escena completa: modelo, materiales PBR, estudio de luz y cámaras de cada vista |
| `modelo/texturas/` | Texturas que usa el `.blend` (rutas relativas) |
| `modelo/olympikus_lumi.glb` | Versión glTF 2.0 con texturas horneadas (color + normal), lista para web / AR |
| `pipeline/` | Código que genera todo (geometría paramétrica, texturas, materiales, estudio) y datos medidos |

## Vistas renderizadas

| Archivo | Vista |
|---|---|
| `perfil_lateral.png` | Perfil lateral completo |
| `perfil_medial.png` | Perfil medial, con la impresión EVASENSE |
| `tres_cuartos_delantera.png` | Tres cuartos delantera |
| `tres_cuartos_trasera.png` | Tres cuartos trasera |
| `cenital_lengueta_interior.png` | Cenital: lengüeta, cordones, plantilla e interior |
| `suela.png` | Suela con su dibujo de tracción y el logotipo grabado |
| `macro_logotipo.png` | Logotipo Olympikus en relieve de TPU con filete menta |
| `macro_etiqueta_lumi.png` | Etiqueta LUMI, cordón tejido y malla de la lengüeta |
| `macro_tejido_jacquard.png` | Punto jacquard del upper |
| `macro_talon.png` | Talón, tirador y cuello acolchado |

## Cómo se construyó

1. **Medición de las fotos** (`pipeline/analisis/`): segmentación del fondo blanco, perfiles superior/inferior,
   línea entresuela–upper, ancho en planta, borde de la ojalera de TPU, 6 ojales por lado y las 7 ondas
   esculpidas de cada pared de la entresuela, todo normalizado al largo de la suela (265 mm).
2. **Geometría paramétrica** (`shoe_geom.py`, `shoe_parts.py`): entresuela por secciones con las ondas
   desplazadas sobre la pared, horma del upper con sección superelíptica y talón polar, cuello acolchado con
   ribete enrollado, lengüeta acolchada, ojalera de TPU con escudo en relieve, ojales, cordones planos tejidos
   cruzados, tirador de talón, plantilla y logotipo en relieve (altura generada a partir del trazado vectorial).
3. **Texturas** (`make_textures.py`): el jacquard se proyecta desde las propias fotos (des-iluminado y con
   relleno que conserva el tejido bajo el logo y la ojalera); logotipo Olympikus vectorizado desde el filete
   menta de la foto; etiqueta LUMI rectificada por homografía; tipografía de `FEITO◇POR BRASILEIROS`,
   `EVASENSE` y `LUMI` recompuesta; mapa de altura de la suela extraído de la foto inferior.
4. **Materiales PBR** (`materials.py`): punto con *sheen* y micro-relieve de pasadas, EVA mate con grano y leve
   subsuperficie, TPU satinado con *clear coat*, cordón con relieve de tejido, forro con terciopelo.
5. **Estudio** (`studio.py`): softboxes superior, principal, relleno, medial y tira de contorno; suelo
   *shadow catcher* compuesto sobre blanco puro, transformación de vista *Standard*, 96 muestras con
   *denoising* OpenImageDenoise.
6. **Suela**: la foto inferior se proyecta como multiplicador de albedo (sombras de canales, paredes de los
   tacos, logotipo grabado) y como relieve (puntos de agarre y bordes de los tacos).

## Fidelidad

Cada render de `comparacion/` usa una cámara reconstruida a partir de la foto correspondiente (ortográfica en
los laterales, perspectiva en la cenital). La coincidencia se mide como intersección sobre unión (IoU) entre
la silueta de la foto y la del render:

| Vista | IoU de silueta |
|---|---|
| Lateral | 94,9 % |
| Medial | 93,5 % |
| Cenital | 96,2 % |

Limitaciones conocidas: el tejido se proyecta desde tres fotos, así que la parte trasera del talón usa un
parche de punto repetido; las zonas que las fotos no muestran (interior del forro, cara inferior de la
lengüeta) se resolvieron con materiales coherentes, no medidos.

## Abrir y re-renderizar

Abre `modelo/olympikus_lumi.blend` en Blender 5.0 o superior. La escena incluye el estudio (softboxes, suelo
*shadow catcher*) y una cámara por vista (`CAM_q34_front`, `CAM_side_lateral`, `CAM_macro_logo`…). Activa la
cámara que quieras y renderiza con F12: la salida es PNG con transparencia y sombra de contacto; los renders
de `renders/` se compusieron sobre blanco puro.

## Regenerar desde el código

```bash
pip install bpy==5.0.1 numpy==1.26.4 "scipy<1.15" opencv-python-headless==4.10.0.84 scikit-image==0.24.0 pillow fonttools brotli
cd pipeline
cp ../modelo/texturas/* tex/                 # texturas ya generadas (o make_textures.py con las fotos)
python build_shoe.py                         # geometría + materiales -> out/lumi.blend
python studio.py -- out/lumi.blend side_lateral,q34_front,top_down,sole 1800 96 renders -0.25
python export_glb.py -- out/lumi.blend out/olympikus_lumi.glb
```

`make_textures.py` y los scripts de `pipeline/analisis/` necesitan las fotos originales en `pipeline/ref/`
(`652LUMILILA_1.jpg` … `_5.jpg`), que no se incluyen en el repositorio. `pipeline/ref/` sí contiene los datos ya
medidos que usa la geometría: perfiles, contornos en planta, borde de la ojalera y líneas de las ondas.
