ORDERED_SOURCES = ['musicgen', 'gtzan']
SOURCE_MAP = {source: i for i, source in enumerate(ORDERED_SOURCES)}

def source_to_label(source):
    return SOURCE_MAP[source]

def label_to_source(label):
    return ORDERED_SOURCES[label]

