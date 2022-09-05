"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F
from models import sparse_token as sparse

class DistillDiffPruningLoss_dynamic(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, teacher_model, base_criterion: torch.nn.Module, ratio_weight=2.0, distill_weight=0.5, dynamic=False, pruning_loc=[3,6,9], keep_ratio=[0.75, 0.5, 0.25], clf_weight=0, mse_token=False, print_mode=True):
        super().__init__()
        self.teacher_model = teacher_model
        self.base_criterion = base_criterion
        self.clf_weight = clf_weight
        self.pruning_loc = pruning_loc
        self.keep_ratio = keep_ratio
        self.count = 0
        self.print_mode = print_mode
        self.cls_loss = 0
        self.ratio_loss = 0
        self.cls_distill_loss = 0
        self.token_distill_loss = 0
        self.mse_token = mse_token
        self.dynamic = dynamic

        self.ratio_weight = ratio_weight
        self.distill_weight = distill_weight

        print('ratio_weight=', ratio_weight, 'distill_weight', distill_weight)


        if dynamic:
            print('using dynamic loss')
    
    def forward(self, batch, output_hf, base_loss, model:"sparse.ApproxSparseBertForSequenceClassification"):
        output_student = output_hf
        with torch.no_grad():
            output_teacher = self.teacher_model(**batch)
        
        # kl(cls_s, cls_t) + kl(last_hid_s, last_hid_t)
        logit_s = output_student.logits
        logit_t = output_teacher.logits
        assert logit_s.shape == logit_t.shape
        assert len(logit_s.shape) == 2
        cls_kl_loss = F.kl_div(
            F.log_softmax(logit_s, dim=-1),
            F.log_softmax(logit_t, dim=-1),
            reduction='batchmean',
            log_target=True
        )

        #remove classification token
        # hidden_s = output_student.hidden_states[:, 1:, :]
        # hidden_t = output_teacher.hidden_states[:, 1:, :]
        # assert hidden_s.shape == hidden_t.shape
        # assert len(hidden_s.shape) == 3

        # mask should be last layer's output mask, except classification token

        # Since we remove all output tokens except classification token in Concrete Masking model,
        # We cannot compute token_kl_loss
        
        loss = base_loss + self.distill_weight * cls_kl_loss
        # + self.distill_weight * token_kl_loss

        # if self.print_mode:
        #     if (self.count % 100) == 0:
        #         print(f"DistillDiffPruningLoss: loss={loss.item():.4f}, loss_base={base_loss.item():.4f}, loss_cls_kl={self.distill_weight * cls_kl_loss.item():.4f}")
        #     self.count += 1

        return loss, {
            'loss': loss.item(),
            'loss_base': base_loss.item(),
            'loss_kl': cls_kl_loss.item(),
        }

    def forward_old(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        pred, token_pred, mask, out_pred_score = outputs

        pred_loss = 0.0

        ratio = self.keep_ratio
        for i, score in enumerate(out_pred_score):
            if self.dynamic:
                pos_ratio = score.mean()
            else:
                pos_ratio = score.mean(1)
            pred_loss = pred_loss + ((pos_ratio - ratio[i]) ** 2).mean()

        cls_loss = self.base_criterion(pred, labels)

        with torch.no_grad():
            cls_t, token_t = self.teacher_model(inputs)

        cls_kl_loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                F.log_softmax(cls_t, dim=-1),
                reduction='batchmean',
                log_target=True
            )

        B, N, C = token_pred.size()
        assert mask.numel() == B * N

        bool_mask = mask.reshape(B*N) > 0.5

        loss_part = []

        token_pred = token_pred.reshape(B*N, C)
        token_t = token_t.reshape(B*N, C)

        if mask.sum() < 0.1:
            token_kl_loss = token_pred.new(1,).fill_(0.0)
        else:
            token_t = token_t[bool_mask]
            token_pred = token_pred[bool_mask]
            if self.mse_token:
                token_kl_loss = torch.pow(token_pred - token_t, 2).mean()
            else:
                token_kl_loss = F.kl_div(
                        F.log_softmax(token_pred, dim=-1),
                        F.log_softmax(token_t, dim=-1),
                        reduction='batchmean',
                        log_target=True
                    )
        
        # print(cls_loss, pred_loss)
        loss = self.clf_weight * cls_loss + self.ratio_weight * pred_loss / len(self.pruning_loc) + self.distill_weight * cls_kl_loss + self.distill_weight * token_kl_loss

        if self.print_mode:
            self.cls_loss += cls_loss.item()
            self.ratio_loss += pred_loss.item()
            self.cls_distill_loss += cls_kl_loss.item()
            self.token_distill_loss += token_kl_loss.item()
            loss_part.append(cls_loss)
            loss_part.append(pred_loss)
            loss_part.append(cls_kl_loss)
            loss_part.append(token_kl_loss)
            self.count += 1
            if self.count == 100:
                print('loss info: cls_loss=%.4f, ratio_loss=%.4f, cls_kl=%.4f, token_kl=%.4f' % (self.cls_loss / 100, self.ratio_loss / 100, self.cls_distill_loss/ 100, self.token_distill_loss/ 100))
                self.count = 0
                self.cls_loss = 0
                self.ratio_loss = 0
                self.cls_distill_loss = 0
                self.token_distill_loss = 0
        return loss, loss_part

