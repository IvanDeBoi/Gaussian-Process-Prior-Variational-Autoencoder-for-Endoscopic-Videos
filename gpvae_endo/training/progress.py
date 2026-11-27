
class ProgressTracker:
    def __init__(
            self,
            patience=1000,
            max_iterations=10000,
            print_iterations=100,
            early_stopping_start_iteration=0,
    ):
        self.best_loss = float('inf')
        self.prev_best_loss = float('inf')
        self.patience = patience
        self.no_improvement_count = 0
        self.max_iterations = max_iterations
        self.print_iterations: int = 1
        self.early_stopping_start_iteration = early_stopping_start_iteration
        self.do_not_print_first_n_iterations = 10
        print(
            f"Initialized progress tracker with patience={patience}, "
            f"early_stopping_start_iteration={self.early_stopping_start_iteration}, "
            f"max_iterations={self.max_iterations}, print_iterations={self.print_iterations}"
        )

    def check_stop_criterion(self, loss, current_iteration):
        must_stop = False
        self.prev_loss = loss.item()

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self.no_improvement_count = 0
        else:
            if current_iteration > self.early_stopping_start_iteration:
                self.no_improvement_count += 1
            else:
                self.no_improvement_count = 0

        if current_iteration > self.max_iterations:
            print(f"  Stopping at iteration {current_iteration}: maximum iterations reached")
            must_stop = True

        if self.no_improvement_count >= self.patience:
            print(f"  Early stopping at iteration {current_iteration}: no improvement for {self.patience} iterations")
            must_stop = True

        # if (current_iteration > self.do_not_print_first_n_iterations and current_iteration % self.print_iterations == 0) or current_iteration == self.max_iterations:
        #     print(f"   Iter {current_iteration}: loss = {loss.item():.2f}, no improvement count = {self.no_improvement_count}")

        return must_stop
