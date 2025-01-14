# LECTURE D'UNE IMAGE
import cv2

# 1/ Charger une image
image = cv2.imread('image.jpg')

# Afficher l'image
#cv2.imshow('Image', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 2/ CONVERSION EN GRIS

# Convertir en niveaux de gris
gray_image = cv2.cvtColor(image,
cv2.COLOR_BGR2GRAY)

# Sauvegarder ou afficher
#cv2.imshow('Grayscale Image', gray_image)
#cv2.imwrite('gray_image.jpg', gray_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 3/ REDIMENTIONNEMENT 
# Redimensionner à une taille fixe
resized_image = cv2.resize(image, (200, 200))

# Redimensionner en maintenant le ratio
height, width = image.shape[:2]
new_width = 300
new_height = int((new_width / width) * height)
resized_aspect_image = cv2.resize(image,
(new_width, new_height))

#cv2.imshow('Resized Image', resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 4/ NORMALISATION 

# Normalisation entre 0 et 1
normalized_image = image / 255.0

# Normalisation à une plage personnalisée
norm_image = cv2.normalize(image, None,
alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow('normalize', norm_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
