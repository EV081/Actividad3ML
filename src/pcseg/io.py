"""Lectura y escritura de PLY binario sin depender de open3d."""

import numpy as np


# Colores tab20, compartidos con las figuras y repetidos cada 20 etiquetas.
palette = np.array([
    [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
    [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
    [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
    [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
    [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
], dtype=np.uint8)
palette.setflags(write=False)

_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "short": "<i2", "ushort": "<u2",
    "int": "<i4", "uint": "<u4", "float": "<f4", "double": "<f8",
    "int8": "i1", "uint8": "u1", "int16": "<i2", "uint16": "<u2",
    "int32": "<i4", "uint32": "<u4", "float32": "<f4", "float64": "<f8",
}


def read_ply(path):
    """Lee vértices binary_little_endian; devuelve xyz, rgb y normals."""
    with open(path, "rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("El archivo no es PLY")
        properties = []
        count = None
        element = None
        binary = False
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("Header PLY incompleto")
            words = line.decode("ascii").split()
            if not words:
                continue
            if words[0] == "format":
                binary = words[1:] == ["binary_little_endian", "1.0"]
            elif words[0] == "element":
                if count is None and words[1] != "vertex" and int(words[2]):
                    raise ValueError("Los vértices deben ser el primer elemento")
                element = words[1]
                if element == "vertex":
                    count = int(words[2])
            elif words[0] == "property" and element == "vertex":
                if len(words) != 3 or words[1] not in _PLY_TYPES:
                    raise ValueError("Propiedad de vértice no soportada")
                properties.append((words[2], _PLY_TYPES[words[1]]))
            elif words[0] == "end_header":
                break
        if not binary or count is None or count < 0:
            raise ValueError("Se requiere PLY binary_little_endian con vértices")
        vertices = np.fromfile(stream, dtype=np.dtype(properties), count=count)
        if len(vertices) != count:
            raise ValueError("Datos PLY truncados")
    names = vertices.dtype.names
    colors = ("r", "g", "b") if "r" in names else ("red", "green", "blue")
    groups = {"xyz": (("x", "y", "z"), np.float32),
              "rgb": (colors, np.uint8),
              "normals": (("nx", "ny", "nz"), np.float32)}
    result = {}
    for key, (fields, dtype) in groups.items():
        if not all(field in names for field in fields):
            if key == "normals":
                result[key] = np.zeros((count, 3), dtype=dtype)
                continue
            raise ValueError(f"Faltan propiedades para {key}")
        result[key] = np.column_stack([vertices[field] for field in fields]).astype(dtype)
    return result


def write_ply(path, xyz, rgb, normals=None):
    """Escribe x y z float, r g b uchar y, si se dan, nx ny nz float."""
    xyz = np.asarray(xyz)
    rgb = np.asarray(rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape:
        raise ValueError("xyz y rgb deben tener forma (N, 3)")
    fields = [(name, "<f4") for name in ("x", "y", "z")]
    fields += [(name, "u1") for name in ("r", "g", "b")]
    columns = [*xyz.T, *rgb.T]
    if normals is not None:
        normals = np.asarray(normals)
        if normals.shape != xyz.shape:
            raise ValueError("normals debe tener forma (N, 3)")
        fields += [(name, "<f4") for name in ("nx", "ny", "nz")]
        columns += list(normals.T)
    vertices = np.empty(len(xyz), dtype=np.dtype(fields))
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    for (name, dtype), values in zip(fields, columns):
        vertices[name] = values
        header.append(f"property {'uchar' if dtype == 'u1' else 'float'} {name}")
    header.append("end_header")
    with open(path, "wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        vertices.tofile(stream)


def labels_to_rgb(labels, palette=None):
    """Colorea etiquetas enteras; -1 siempre es gris oscuro (45, 45, 45)."""
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels debe ser un vector de enteros")
    colors = globals()["palette"] if palette is None else np.asarray(palette)
    if colors.ndim != 2 or colors.shape[1] != 3 or len(colors) == 0:
        raise ValueError("palette debe tener forma (K, 3), K > 0")
    rgb = np.asarray(colors[labels % len(colors)], dtype=np.uint8).copy()
    rgb[labels == -1] = 45
    return rgb
