import cv2
from PIL import Image
import matplotlib.pyplot as plt
from segmentation import segment_chars

def get_processed_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = thresh
    return ~result


def predict_chars(rect_pos, rects, model, k=2, orig_img=None, debug=False):
    res = []
    for inputs in rects:
        inputs = cv2.resize(inputs, (W, H))
        inputs = (torch.Tensor(inputs).float() / 255.).reshape(1, 1, *inputs.shape)
        inputs = 1 - inputs
        _, inds = torch.topk(model(inputs), k=k, sorted=True)
        res.append(inds.numpy())
    res = [tuple(map(lambda c: label_map[allowed_chars[c]], x[0])) for x in res]
    if debug:
        new_img = np.copy(orig_img)
        for i, (x, y, w, h) in enumerate(rect_pos):
            new_img = cv2.putText(new_img, res[i][0], (x+w, y), cv2.FONT_HERSHEY_COMPLEX, 
                                  .4, (0, 255, 0))
        
        plt.figure(figsize=(20, 20))
        plt.imshow(new_img)
    return res


def analyze(img):
    processed_img = get_processed_img(img)
    rect_pos, rects = segment_chars(processed_img, debug=False)
    predict_chars(rect_pos, rects, model, k=2, debug=True, orig_img=processed_img);
