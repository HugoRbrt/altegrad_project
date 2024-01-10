# add here other functions to help you train your model
# example : way to compute the accuracy...

def is_correct(pred, label):
    """_summary_

    Args:
        pred (_type_): list of similarities
        label (_type_): _description_

    Returns:
        _type_: _description_
    """
    return (pred == label).sum().item()