import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List


def catarata_mask_selector(
    image_path: str,
    k: int,
    proto: Dict[int, Dict[str, Any]],
    mask_generator,
    get_emb,
    visualize: bool = True,
) -> Tuple[str, Optional[np.ndarray]]:
    """Analiza las máscaras generadas por SAM y devuelve la mejor máscara
    únicamente si corresponde a *catarata*.

    Parameters
    ----------
    image_path : str
        Ruta del archivo de imagen a procesar.
    k : int
        Índice del prototipo (p.ej. *k*-shot) que se evaluará.
    proto : dict
        Diccionario con estadísticas del prototipo. Debe contener las claves
        ``kdes``, ``theta_min`` y ``theta_max`` para cada *k*.
    mask_generator : sam.MaskGeneratorLike
        Instancia de MaskGenerator (o clase compatible) que ofrece el método
        ``generate(image_np)`` y devuelve una lista de máscaras (dict).
    get_emb : Callable[[PIL.Image.Image], np.ndarray]
        Función que recibe un recorte (crop) y devuelve el embedding 1‑D
        correspondiente.
    visualize : bool, default True
        Si ``True`` dibuja las máscaras evaluadas y la mejor máscara
        seleccionada. Si ``False`` la función trabaja en segundo plano.

    Returns
    -------
    Tuple[str, Optional[np.ndarray]]
        ``("Catarata", best_mask_np)``    si alguna máscara cae dentro del
        rango [θ_min, θ_max]. ``best_mask_np`` es un *array* binario de la
        segmentación seleccionada (shape = H×W) que puede emplearse como
        *ground‑truth* o visualizarse posteriormente.

        ``("No Catarata", None)``         si *ninguna* máscara cumple el
        criterio; en este caso **no** se devuelve máscara alguna.
    """

    # 1) Carga imagen
    img_pil: Image.Image = Image.open(image_path).convert("RGB")
    img_np: np.ndarray = np.asarray(img_pil)

    # 2) Genera máscaras automáticas con SAM/AMG
    masks: List[Dict[str, Any]] = mask_generator.generate(img_np)

    # 3) Recupera estadísticos de KDE para este *k*
    kdes = proto[k]["kdes"]
    t_min = proto[k]["theta_min"]
    t_max = proto[k]["theta_max"]

    scores: List[float] = []
    preds: List[int] = []

    # 4) Evalúa cada máscara
    for i, m in enumerate(masks):
        ys, xs = np.where(m["segmentation"])
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        crop = img_pil.crop((x0, y0, x1 + 1, y1 + 1))
        emb = get_emb(crop)

        # 4.1) Log‑score usando el KDE de *cada* dimensión
        logp = sum(kde.score_samples([[emb[d]]])[0] for d, kde in enumerate(kdes))
        scores.append(logp)

        # 4.2) Clasificación binaria: 1 = dentro del rango (catarata)
        preds.append(int(t_min <= logp <= t_max))

        # 4.3) Visualización individual (opcional)
        if visualize:
            plt.figure(figsize=(6, 6))
            plt.imshow(img_np)
            poly = patches.Polygon(
                np.column_stack([xs, ys]),
                closed=True,
                facecolor=(0.8, 0.8, 0.8, 0.4),
                edgecolor="black",
            )
            plt.gca().add_patch(poly)
            plt.text(
                x0,
                y0 - 10,
                f"Máscara #{i}\nlogp = {logp:.2f}\n→ {'CAT' if preds[-1] else 'NO'}",
                color="white",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.6, pad=4),
            )
            plt.axis("off")
            plt.title(f"Máscara {i + 1}/{len(masks)}")
            plt.show()

    # 5) Selecciona la mejor máscara dentro de los positivos; si no hay, None
    positives = [i for i, p in enumerate(preds) if p == 1]
    if positives:
        # Entre las positivas, la de mayor log‑probabilidad
        best_idx = max(positives, key=lambda i: scores[i])
        result_label = "Catarata"
    else:
        # Ninguna máscara fue considerada catarata
        if visualize:
            print("No Catarata — ninguna máscara cayó dentro del rango especificado.")
        return "No Catarata", None

    # 6) Extrae máscara binaria de la mejor selección
    best_mask_np: np.ndarray = masks[best_idx]["segmentation"].astype(np.uint8)

    return result_label, best_mask_np