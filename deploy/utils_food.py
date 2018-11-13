import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing import image

def plot_preds(image, probabilities, top_n, categories):
    """Plot the probabilities from the Inception network."""
    plt.imshow(image)
    plt.axis('off')
    plt.figure()

    order = list(reversed(range(top_n)))
    labels = [categories[x] for i, x in enumerate(np.argsort(-probabilities)[:top_n])]
    bar_preds = [-np.sort(-probabilities)[i] for i, x in enumerate(np.argsort(-probabilities)[:top_n])]

    plt.barh(order, bar_preds, alpha=0.8, color='g')
    plt.title('the higher, the better', loc='center')
    plt.yticks(order, labels, color='g')
    plt.xlabel('Probability', color='g')
    plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()

def plot_distances(distances, idx_closest, images):
    """Plot the distances from ANN.""" 
    order = list(range(len(distances)))[::-1] # just a list from 0 to self.k-1
    labels = [str(images[i]).split('/')[1] for i in idx_closest]
    
    plt.barh(order, distances, alpha=0.8, color='g')
    plt.title('the lower, the better', loc='center')
    plt.yticks(order, labels, color='g')
    plt.xlabel('L2 distance (euclidean)', color='g')
    plt.tight_layout()
    plt.show()

def show_result_images(final_result):
    """Show only the images from the resulting food list."""
    rows = 2
    cols = 3
    fig, ax = plt.subplots(rows, cols, frameon=False, figsize=(12, 12))
    fig.suptitle('Result Images from Query', fontsize=20)
    food_dirs = [food_direction[4] for food_direction in final_result]
    for i in range(rows):
      for j in range(cols):
        food_dir = food_dirs[i*cols + j]
        img = plt.imread(food_dir)
        ax[i][j].imshow(img)
        ec = (0, .6, .1)
        fc = (1, 1, 1)
        ax[i][j].text(0, 0, get_corresponding_recipes(final_result).recipe_name[i*cols + j], size=15, rotation=0,
                ha="left", va="top", 
                bbox=dict(boxstyle="round", ec=ec, fc=fc))
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def get_concatenated_images(indexes, thumb_height):
    """Return a concated image of multiple food images from local disk."""
    thumbs = []
    for idx in indexes:
        img = image.load_img(images[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image

def get_corresponding_recipes(final_results, chefkoch_rezepte_csv):
    chefkoch_rezepte_result = pd.DataFrame()
    for recipe_id_single in final_results:
        chefkoch_rezepte_result = chefkoch_rezepte_result.append(chefkoch_rezepte_csv.query('recipe_id in @recipe_id_single[2]'), ignore_index=True)
    return chefkoch_rezepte_result
