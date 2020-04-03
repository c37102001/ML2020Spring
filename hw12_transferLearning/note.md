# Accuracy
| Model                 | Public Acc    | Private Acc   |
| --------------------- | ------------- | ------------- |
| baseline              | 0.52810       |               |
| baseline w/o canny    | 0.56355       |               |


# Misc
loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)

loss中 - lamb * domain_criterion(domain_logits, domain_label)
如果把domain_label倒過來，讓domain=0的在算domain_criterion時label給1，domain=1時給0，是否可以達到同樣的效果，
前面的減號就可改成加號?