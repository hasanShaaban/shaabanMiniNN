import numpy as np

def eval_loss(model, loss_fn, X, y):
    scores = model.forward(X)
    return loss_fn.forward(scores, y)

def numerical_gradient(model, loss_fn, X, y, eps=1e-5):
    num_grads = []

    for params,_ in model.params_and_grads():
        grads = {}
        for key in params :
            grad = np.zeros_like(params[key])
            it = np.nditer(params[key], flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                idx = it.multi_index
                original = params[key][idx]

                params[key][idx] = original+eps
                loss_plus = eval_loss(model, loss_fn, X, y)

                params[key][idx] = original-eps
                loss_minus = eval_loss(model, loss_fn, X, y)

                params[key][idx] = original

                grad[idx] = (loss_plus - loss_minus) /(2 * eps)
                it.iternext()

            grads[key] = grad
        num_grads.append(grads)
    return num_grads

def gradient_check(model, loss_fn, X, y, eps=1e-5, tol=1e-7):
    scores = model.forward(X)
    loss_fn.forward(scores, y)
    dout = loss_fn.backward()
    model.backward(dout)

    analytical = []

    for _, grads in model.params_and_grads():
        analytical.append({k:v.copy() for k,v in grads.items()})

    num_grads = numerical_gradient(model, loss_fn, X, y, eps)

    for ana, num in zip(analytical, num_grads):
        for key in ana:
            diff = np.linalg.norm(ana[key] - num[key])
            denom = np.linalg.norm(ana[key]) + np.linalg.norm(num[key])
            rel_error = diff / (denom + 1e-12)

            print(
                f"{key}: "
                f"error = {rel_error: .2e}"
            )

            if rel_error > tol: 
                print('gradient check failed')
                return

    print('gradient check passed')


