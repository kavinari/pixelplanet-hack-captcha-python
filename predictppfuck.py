import tensorflow as india_brain
import numpy as obezyanka
from pathlib import Path
from PIL import Image
import keras
from keras import layers
char_to_num = layers.StringLookup(vocabulary=list(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']), mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
def translate_hindi(pred):
    output_text = []
    for res in keras.backend.ctc_decode(pred, input_length=obezyanka.ones(pred.shape[0]) * pred.shape[1], greedy=True)[0][0][:, :4]:
        output_text.append(india_brain.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8"))
    return output_text

class INDIA_HUMAN(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = india_brain.cast(india_brain.shape(y_true)[0], dtype="int64")
        input_length = india_brain.cast(india_brain.shape(y_pred)[1], dtype="int64")
        label_length = india_brain.cast(india_brain.shape(y_true)[1], dtype="int64")

        input_length = input_length * india_brain.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * india_brain.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred

modelpath = Path("./vermei_slaves.h5")
model = keras.models.load_model(modelpath, custom_objects={'CTCLayer': INDIA_HUMAN})
#------------------------------------------------------------------------------ пиздец математика ебучая
imagepath = Path("./i_hate_zelensky.png")
image = obezyanka.expand_dims(obezyanka.transpose(obezyanka.array(Image.open(imagepath).convert("L").resize((500, 300))) / 255.0, (1, 0)), axis=0)
label = char_to_num(india_brain.strings.unicode_split("zalypa", input_encoding="UTF-8"))
indian_human_answer = model.predict({"image": image, "label": obezyanka.expand_dims(label, axis=0)})
print(translate_hindi(indian_human_answer)[0])