# Copyright (c) 2020, Zhouxing shi <zhouxingshichn@gmail.com>
# Licenced under the BSD 2-Clause License.

import torch
import math, time, random, copy, json, pdb
from Verifiers.Bounds import Bounds
from data_utils import sample

# can only accept one example in each batch
def project_onto_p_ball(delta, p, radius):
    if p == float('inf'):
        # p = ∞
        return torch.clamp(delta, -radius, radius)
    elif p == 2:
        # p = 2
        delta_norm = torch.norm(delta, p=2, dim=(1, 2), keepdim=True)
        return torch.where(delta_norm > radius, delta / delta_norm * radius, delta)
    else:
        delta_flat = delta.view(delta.size(0), -1)  # Flatten across seq_len and embed_dim
        delta_norm = torch.norm(delta_flat, p=p, dim=1, keepdim=True)  # p-norm
        scaling_factor = torch.where(delta_norm > radius, radius / delta_norm, torch.ones_like(delta_norm))
        delta_flat = delta_flat * scaling_factor  # Scale to the p-norm ball
        return delta_flat.view_as(delta)  # Reshape back to original shape

# can only accept one example in each batch
class Verifier:
    def __init__(self, args, target, logger):
        self.args = args
        self.device = args.device
        #self.device=torch.device("cpu")
        print(self.device)
        self.target = target
        self.logger = logger
        self.res = args.res
        self.p = args.p if args.p < 10 else float("inf")
        self.eps = args.eps
        self.debug = args.debug
        self.verbose = args.debug or args.verbose 
        self.method = args.method
        self.num_verify_iters = args.num_verify_iters
        self.max_eps = args.max_eps
        self.debug_pos = args.debug_pos
        self.perturbed_words = args.perturbed_words
        self.warmed = False

        self.embeddings = target.model.bert.embeddings
        self.encoding_layers = target.model.bert.encoder.layer
        self.pooler = target.model.bert.pooler
        self.classifier = target.model.classifier
        self.hidden_act = args.hidden_act
        self.layer_norm = target.model.config.layer_norm\
            if hasattr(target.model.config, "layer_norm") else "standard"

    def run(self, data):
        examples = sample(self.args, data, self.target) 
        print("{} valid examples".format(len(examples)))
        sum_avg, sum_min = 0, 0
        results = []
        for i, example in enumerate(examples):
            self.logger.write("Sample", i)
            res = self.verify(example)
            if self.debug: 
                continue
            results.append(res[0])
            sum_avg += res[1]
            sum_min += res[2]

        self.logger.write("{} valid examples".format(len(examples)))
        self.logger.write("Minimum: {:.5f}".format(float(sum_min) / len(examples)))
        self.logger.write("Average: {:.5f}".format(float(sum_avg) / len(examples)))
        for i, example in enumerate(examples):
            RE=results[i]
            for r in RE["bounds"]:
                print("({},{:.5f})".format(r["position"], r["eps"]))
        result = {
            "examples": results,
            "minimum": float(sum_min) / len(examples),
            "average": float(sum_avg) / len(examples)
        }
        with open(self.res, "w") as file:
            file.write(json.dumps(result, indent=4))


    def verify(self, example):
        start_time = time.time()        
        
        embeddings, tokens = self.target.get_embeddings([example])
        length = embeddings.shape[1]
        tokens = tokens[0]

        self.logger.write("tokens:", " ".join(tokens))
        self.logger.write("length:", length)        
        self.logger.write("label:", example["label"])

        self.std = self.target.step([example])[-1] 

        result = {
            "tokens": tokens,
            "label": float(example["label"]),
            "bounds": []
        }

        if self.debug:
            eps = self.eps
            index = self.debug_pos
            safety = self.verify_safety(example, embeddings, index, self.eps)  
            self.logger.write("Time elapsed", time.time() - start_time)
            return eps
        else:
            #self.attack(example)
            eps = torch.zeros(length)
            num_iters = self.num_verify_iters

            cnt = 0
            sum_eps, min_eps = 0, 1e30

            # TODO: redundant
            if self.perturbed_words == 1:
                #####test###
                #[0.25, 0.30, 0.40, 0.45]  #0.28
                # for eps in[0.37854,0.37884]:   #0.40,0.39,0.38,0.37,0.36,0.35,0.34,0.33,0.32,0.31,0.30 0.025 0.33572
                #     safe=self.verify_safety(example, embeddings, [1], eps) #[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18] #
                #     print("eps:{:.7f} Safe:{}".format(eps,safe))
                # exit(0)
                ##########################
                # warm up 
                if not self.warmed:
                    print("Warming up...")
                    while not self.verify_safety(example, embeddings, [1], self.max_eps):
                        self.max_eps /= 2
                    while self.verify_safety(example, embeddings, [1], self.max_eps):
                        self.max_eps *= 2
                    self.warmed = True
                    print("Approximate maximum eps:", self.max_eps)
                # [CLS] and [SEP] cannot be perturbed
                for i in range(1, 4): #(1, length - 1)
                    # skip OOV
                    if tokens[i][0] == "#" or tokens[i + 1][0] == "#":
                        continue

                    cnt += 1

                    l, r = 0, self.max_eps
                    print("{} {:.5f} {:.5f}".format(i, l, r), end="\n")
                    safe = self.verify_safety(example, embeddings, [i], r)
                    while safe: 
                        l = r
                        r *= 2
                        print("\r{} {:.5f} {:.5f}".format(i, l, r), end="\n")
                        safe = self.verify_safety(example, embeddings, [i], r)
                    if l == 0:
                        while not safe:
                            r /= 2
                            print("\r{} {:.5f} {:.5f}".format([i], l, r), end="\n")
                            safe = self.verify_safety(example, embeddings, [i], r)
                        l, r = r, r * 2
                        print("\r{} {:.5f} {:.5f}".format([i], l, r), end="\n")
                    for j in range(num_iters):
                        m = (l + r) / 2
                        if self.verify_safety(example, embeddings, [i], m):
                            l = m
                        else:
                            r = m
                        print("\r{} {:.5f} {:.5f}".format([i], l, r), end="\n")
                    #print()
                    eps[i] = l
                    self.logger.write("Position {}: {} {:.5f}".format(
                        i, tokens[i], eps[i], ))
                    sum_eps += eps[i]
                    min_eps = min(min_eps, eps[i])
                    norm = torch.norm(embeddings[0, [i], :], p=self.p)
                    result["bounds"].append({
                        "position": i,
                        "eps": float(eps[i]),
                        "eps_normalized": float(eps[i] / norm)
                    })

            elif self.perturbed_words == 2:
                # warm up 
                if not self.warmed:
                    print("Warming up...")
                    while not self.verify_safety(example, embeddings, [1, 2], self.max_eps):
                        self.max_eps /= 2
                    while self.verify_safety(example, embeddings, [1, 2], self.max_eps):
                        self.max_eps *= 2
                    self.warmed = True
                    print("Approximate maximum eps:", self.max_eps)

                for i1 in range(1, length - 1):
                    for i2 in range(i1 + 1, length - 1):
                        # skip OOV
                        if tokens[i1][0] == "#" or tokens[i1 + 1][0] == "#":
                            continue
                        if tokens[i2][0] == "#" or tokens[i2 + 1][0] == "#":
                            continue                            

                        cnt += 1

                        l, r = 0, self.max_eps
                        print("%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                        safe = self.verify_safety(example, embeddings, [i1, i2], r)
                        while safe: 
                            l = r
                            r *= 2
                            print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                            safe = self.verify_safety(example, embeddings, [i1, i2], r)
                        if l == 0:
                            while not safe:
                                r /= 2
                                print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                                safe = self.verify_safety(example, embeddings, [i1, i2], r)
                            l, r = r, r * 2
                            print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                        for j in range(num_iters):
                            m = (l + r) / 2
                            if self.verify_safety(example, embeddings, [i1, i2], m):
                                l = m
                            else:
                                r = m
                            print("\r%d %d %.6f %.6f" % (i1, i2, l, r), end="")
                        print()
                        eps = l
                        self.logger.write("Position %d %d: %s %s %.5f" % (
                            i1, i2, tokens[i1], tokens[i2], eps))
                        sum_eps += eps
                        min_eps = min(min_eps, eps)
                        result["bounds"].append({
                            "position": (i1, i2),
                            "eps": float(eps)
                        })                        
            else:
                raise NotImplementedError

            result["time"] = time.time() - start_time

            self.logger.write("Time elapsed", result["time"])
            return result, sum_eps / cnt, min_eps

    def attack(self,example,att_embeddings,radius,index,it_num):
        att_embeddings.requires_grad_()
        delta = torch.zeros_like(att_embeddings)
        step = radius/7.0
        delta_mask = torch.zeros_like(att_embeddings)
        delta_mask[:,index,:] = 1

        '''
        ret = [
            loss, acc,
            {
                "pred_scores": preds, 
                "pred_labels": pred_labels,
                "embedding_output": embedding_output,
                "encoded_layers": encoded_layers,
                "attention_scores": attention_scores,
                "attention_probs": attention_probs,
                "self_output": self_output,
                "pooled_output": pooled_output,
                "all_QK_output": QK_output,
                "features": features,
                "gradients": gradients
            }
        '''
        last_loss=None
        while it_num>0:

            ret = self.target.step([example], infer_grad=True,before_norm_embeddings=(att_embeddings,delta))
            #print(ret[-1]["logits"])
            grad =ret[-1]["gradients"]

            with torch.no_grad():
                #grad = att_embeddings.grad
                if ret[-1]["pred_labels"] != example["label"]:
                    print("attack succesfully!")
                    return True, None
                if last_loss is None:
                    last_loss = ret[0]
                else:
                    if torch.abs((ret[0] - last_loss)) < 1e-5:
                        it_num = 0
                        continue
                last_loss = ret[0]
                delta = delta + step * grad * delta_mask
                delta = project_onto_p_ball(delta, self.p, radius)

            att_embeddings.requires_grad_()
            it_num -= 1

        temp_std1=self.target.step([example],before_norm_embeddings=(att_embeddings,delta))[-1]
        #temp_std1=self.target.step([example])[-1]
        all_QK_output1 = temp_std1["all_QK_output"]
        return False, all_QK_output1

    def findMaxMin(self,example,att_embeddings,radius,index,it_num):

        min_std=None
        max_std=None
        delta_mask = torch.zeros_like(att_embeddings)
        delta_mask[:, index, :] += 1
        for i in range(2):
            att_embeddings.requires_grad_()
            delta = torch.zeros_like(att_embeddings)
            step = radius / 7.0
            last_loss = None
            it = it_num
            while it>0:
                ret = self.target.step([example], infer_grad=True,before_norm_embeddings=(att_embeddings,delta))
                if i ==0:
                    min_std=ret[-1]
                    grad = min_std["gradients"]
                else:
                    max_std=ret[-1]
                    grad = max_std["gradients"]
                with torch.no_grad():
                    if last_loss is None:
                        last_loss = ret[0]
                    else:
                        if torch.abs((ret[0] - last_loss)) < 1e-5:
                            it = 0
                            continue
                        last_loss = ret[0]
                    if i==0:
                        delta = delta + step * grad * delta_mask
                    else:
                        delta = delta - step * grad * delta_mask
                    delta = project_onto_p_ball(delta, self.p, radius)
                att_embeddings.requires_grad_()
                it -= 1

        assert min_std is not None, "min_std is None"
        assert max_std is not None, "min_std is None"

        return min_std, max_std



    def verify_safety(self, example, embeddings, index, eps):
        raise NotImplementedError
