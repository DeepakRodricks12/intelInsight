Adversarial Attacks and Defenses in Deep Reinforcement Learning: A Comprehensive Evaluation in a Highway Environment

Project Description:

This thesis will explore the robustness of Deep Reinforcement Learning (DRL) algorithms (PPO, DQN, DDPG, A3C) when subjected to adversarial attacks within the highway environment simulation. The focus is on real-world tasks like autonomous driving where the agent must navigate through traffic under adversarial conditions. The attacks will focus on manipulating observations, actions, and rewards to study how they degrade performance, and the defenses will aim to make the agent resilient in these environments. Key goals for the project include:
• Implementing Attacks: Fast Gradient Sign Method (FGSM), Projected Gradient Descent(PGD), and Jacobian-based Saliency Map Attack (JSMA) will be used to perturb observations and actions in the highway environment.
• Developing Defenses: Adversarial training, input noise injection, and certified robustness methods will be implemented to protect the agent from these attacks.
• Algorithm Comparison: PPO, DQN, and A3C will be compared for their ability to resist adversarial attacks.
• Benchmarking: Performance will be evaluated in Open AI’s Highway-Env, and resilience benchmarks will be established based on how each algorithm handles the adversarial conditions

Research Methodology:
• Attacks: Implement FGSM, PGD, and JSMA to manipulate observations and actions. Focus on how these affect the agent's ability to drive efficiently in traffic, avoid collisions, and navigate intersections.
• Defenses: Develop and evaluate defenses that make the agent resilient to adversarial conditions. This includes adversarial training by integrating adversarial examples during learning, and input filtering to reduce the impact of adversarial perturbations in real-time decision-making.
• Algorithms: Train the selected algorithms (PPO, DQN, and A3C) in the highway environment, assessing their performance under normal and adversarial conditions.
• Simulations: Use a highway simulation environment (e.g., Highway-env) to simulate realworld scenarios like lane-keeping, car-following, and collision avoidance. Measure key metrics like reward degradation, success rates, and attack success rates.
• Evaluation: Evaluate how different algorithms handle attacks. Compare them based on metrics such as resilience (performance drop under attack), computational efficiency, and adaptability to real-time defenses.
