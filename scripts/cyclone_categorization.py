def categorizeCyclone(intensity):
    category = ''

    if 0 < intensity < 74:
        category = "Tropical Depression"
    elif 74 <= intensity and intensity < 95:
        category = "Tropical Storm"
    elif 95 <= intensity and intensity < 111:
        category = "Category 1"
    elif 111 <= intensity and intensity < 130:
        category = "Category 2"
    elif 130 <= intensity and intensity < 157:
        category = "Category 3"
    elif 157 <= intensity < 178:
        category = "Category 4"
    else:
        category = "Category 5"

    return category
