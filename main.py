import cv2
from PIL import Image
import matplotlib.pyplot as plt


def get_processed_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    result = thresh
    return ~result


def segment_chars(orig_img, debug=False):
    contours, hierarchy = cv2.findContours(orig_img, 1, 2)
    new_img = np.copy(orig_img)
    rects, rect_pos = [], []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        aspect_ratio = w * 1./h
        area = w * h
        if (h < 5) or (w < 5) or (aspect_ratio > 2):
            continue
        rect_pos.append((x, y, w, h))
        rects.append(orig_img[y:y+h, x:x+w])
        if debug:
            new_img = cv2.rectangle(new_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if debug:
        plt.figure(figsize=(20, 20))
        plt.imshow(new_img)
    
    return rect_pos, rects


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
