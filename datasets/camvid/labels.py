def get_classes(label_file_path):
    with open(label_file_path) as f:
        labels = f.readlines()
    labels = [ label.split() for label in labels ]
    colors = list()
    color2id = dict()
    for ix,label in enumerate(labels):
        r, g, b, class_name = label
        labels[ix] = class_name
        colors[ix] = (int(r),int(g),int(b))
        color2id[(int(r),int(g),int(b))] = ix
    return labels, colors, color2id