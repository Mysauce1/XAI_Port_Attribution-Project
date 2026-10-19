# Selected methodological references

These references ground the project design. They do not imply that the selected methods are automatically valid for this dataset; the experiments should evaluate their behavior.

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). **Axiomatic Attribution for Deep Networks.** *Proceedings of the 34th International Conference on Machine Learning (ICML).* Introduces Integrated Gradients and its attribution axioms. https://proceedings.mlr.press/v70/sundararajan17a.html
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). **"Why Should I Trust You?": Explaining the Predictions of Any Classifier.** *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.* Introduces LIME as a local model-agnostic explanation method. https://doi.org/10.1145/2939672.2939778
3. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). **Sanity Checks for Saliency Maps.** *Advances in Neural Information Processing Systems 31.* Discusses tests for whether saliency methods are sensitive to model parameters and learned behavior. https://papers.nips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
4. Doshi-Velez, F., & Kim, B. (2017). **Towards A Rigorous Science of Interpretable Machine Learning.** arXiv:1702.08608. Discusses evaluation challenges for interpretability methods and the importance of task-appropriate evidence. https://arxiv.org/abs/1702.08608
5. DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., & Wallace, B. C. (2020). **ERASER: A Benchmark to Evaluate Rationalized NLP Models.** *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.* Provides a framework for evaluating evidence-based explanations in NLP. https://aclanthology.org/2020.acl-main.408/
6. Hutto, C. J., & Gilbert, E. (2014). **VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.** *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media (ICWSM-14).* The lexicon-and-rules sentiment scorer used in the separate sentiment analysis.

## How these references inform the project

- Integrated Gradients is used as a principled first token-attribution method, with its baseline dependence documented.
- Perturbation/faithfulness checks and stability checks are included because attractive highlights alone do not establish explanation quality.
- Rule traces are treated as a transparent account of program execution, not evidence that the rule's assumptions are normatively or semantically correct.
- The evaluation combines aggregate metrics with case inspection; it does not claim to fully measure human interpretability without a human study.
- VADER is used as a transparent, untrained sentiment baseline, evaluated against the gold sentiment labels rather than assumed correct.
