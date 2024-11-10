ORDERED_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
GENRE_MAP = {genre: i for i, genre in enumerate(ORDERED_GENRES)}

def genre_to_label(genre):
    return GENRE_MAP[genre]

def label_to_genre(label):
    return ORDERED_GENRES[label]

def ordered_genres():
    return [x for x in ORDERED_GENRES]

def genre_to_caption(genre):
    return f"This audio is a {genre} song."

def label_to_caption(label):
    genre = label_to_genre(label)
    return genre_to_caption(genre)

def num_genres():
    return len(ORDERED_GENRES)