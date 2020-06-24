import torch
from torch.autograd import Variable

def compute_saliency_maps(X, y, model):

    model.eval()
    X_var = Variable(X, requires_grad=True)

    loss = torch.nn.MSELoss()
    y_pred = model(X_var)
    output = loss(y_pred, y_var)
    output.backward()
    saliency = torch.abs(X_var.grad)
    saliency, _ = torch.max(saliency, 1)  # over dimension C
    saliency = saliency.data
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()

if __name__ == '__main__':
    pass