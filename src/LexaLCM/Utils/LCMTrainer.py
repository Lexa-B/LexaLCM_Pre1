from transformers import Trainer, Adafactor, get_scheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import wandb

class LCMTrainer(Trainer):
    def __init__(self, *args, config_dict=None, **kwargs):
        self.config_dict = config_dict or {}
        kwargs.pop("config_dict", None)
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_name = self.args.optim.lower()
        params = self.model.parameters()

        if opt_name == "adafactor":
            clip_val = self.config_dict["training"].get("clip_threshold", 1.0)
            self.optimizer = Adafactor(
                params,
                scale_parameter=True,
                relative_step=True,
                clip_threshold=clip_val,
                warmup_init=True,
                # DO NOT pass `lr`
            )

            # 🩹 Manually patch param groups so dummy schedulers don't explode
            for group in self.optimizer.param_groups:
                group["lr"] = 1e-3  # Safe dummy value

            print("[LCMTrainer] ✅ Adafactor initialized with dummy lr for scheduler safety")

        elif opt_name == "adamw":
            self.optimizer = AdamW(
                params,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is not None:
            return self.lr_scheduler

        if optimizer is None:
            optimizer = self.optimizer

        if isinstance(optimizer, Adafactor):
            # ✅ Safe dummy scheduler just to satisfy Hugging Face's `.get_last_lr()`
            self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        else:
            self.lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )

        return self.lr_scheduler

    def log_gradient_norms(self):
        total_norm = 0.0
        grad_report = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                wandb.log({f"grad_norms/{name}": grad_norm}, step=self.state.global_step)
                total_norm += grad_norm ** 2

                # 🖨️ Collect a few gradients to print
                if len(grad_report) < 5:
                    grad_report.append(f"{name}: {grad_norm:.4f}")

        total_norm = total_norm ** 0.5
        wandb.log({"global_grad_norm": total_norm}, step=self.state.global_step)

        # 🖨️ Print the first few gradients once in a while
        if self.state.global_step < 5 or self.state.global_step % 50 == 0:
            print(f"[Gradients @ step {self.state.global_step}] global norm = {total_norm:.4f}")
            for line in grad_report:
                print(f"  • {line}")


    def training_step(self, model, inputs, batch_size=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        loss.backward()

        # ✅ Log gradient norms
        self.log_gradient_norms()

        return loss.detach()