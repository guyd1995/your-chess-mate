
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

