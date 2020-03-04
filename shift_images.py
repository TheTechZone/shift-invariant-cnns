import torch

def shift_h(im, shift_amount):
    """Shift the image horizontally by shift_amount pixels, 
    use positive numbers to shift the image to the right,
    use negative numbers to shift the image to the left"""
    if shift_amount == 0:
        return im
    else:
        if len(im.shape) == 3: # for a single image
            new_image = torch.zeros_like(im)
            new_image[:, :, :shift_amount] = im[:,:,-shift_amount:]
            new_image[:,:,shift_amount:] = im[:,:,:-shift_amount]
            return new_image
        elif len(im.shape) == 4: # for batches of images
            new_image = torch.zeros_like(im)
            new_image[:, :, :, :shift_amount] = im[:, :, :, -shift_amount:]
            new_image[:, :, :, shift_amount:] = im[:, :, :, :-shift_amount]
            return new_image
        

def shift_v(im, shift_amount):
    """Shift the image vertically by shift_amount pixels, 
    use positive number to shift the image down, 
    use negative numbers to shift the image up"""
    if shift_amount == 0:
        return im
    else:
        if len(im.shape) == 3: # for a single image
            new_image = torch.zeros_like(im)
            new_image[:, :shift_amount, :] = im[:,-shift_amount:,:]
            new_image[:,shift_amount:,:] = im[:,:-shift_amount,:]
            return new_image
        elif len(im.shape) == 4: # for batches of images
            new_image = torch.zeros_like(im)
            new_image[:, :, :shift_amount, :] = im[:, :, -shift_amount:, :]
            new_image[:, :, shift_amount:, :] = im[:, :, :-shift_amount, :]
            return new_image


def shift_d(im, shift_amount, dir='br'):
    """Shift the image shift_amount pixels diagonally with, 
    use dir=br to move it to the bottom right,
    use dir=tr to move it to the top right, 
    and use negative numbers to move it in the other direction."""
    if shift_amount == 0:
        return im
    elif dir == 'br':
        return shift_h(shift_v(im, shift_amount), shift_amount)
    elif dir == 'tr':
        return shift_h(shift_v(im,-shift_amount), shift_amount)