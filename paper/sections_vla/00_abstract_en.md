Vision-language-action (VLA) policies generalize through large-scale demonstrations. Yet field data are dominated by nominal driving, so policies remain fragile in rare but safety-critical situations. Vehicles and mobile robots must decide what to keep under storage and bandwidth budgets. Existing robot-data curation, however, assumes a fully collected dataset and a trained policy, and edge-side selection methods are rarely validated by downstream policy performance.

We propose CARE, a collection-time, on-device, policy-agnostic curation method. CARE has three parts:

- A lightweight multichannel CNN-GRU interprets detector-based semantic features and inverse time-to-collision cues, and outputs a per-frame risk probability and uncertainty.
- A budgeted selector mixes top-scoring clips with a random reservoir.
- Instructions, physics-grounded narrations, and action chunks are aligned automatically and exported in the LeRobot format.

We measure data value by the closed-loop success of a small VLA policy (images, instruction, and speed to an acceleration chunk) trained on the selected data. The policy drives the ego vehicle in simulation. Training compute, selection size, and evaluation seeds are held fixed.

At a 2% budget, CARE reaches a closed-loop success rate of {{s:driving:ours:0.02:success}}, versus {{s:driving:random:0.02:success}} for random selection (paired difference {{b:ours_vs_random_b0.02}}). Methods that maximize hazardous samples suffer a "risk-bias collapse", in which the policy brakes everywhere; their success rates stay at or below {{s:driving:oracle:0.02:success:2}}. These methods include ego-deceleration triggers, a hazard-label oracle, and policy-loss-based offline selection. Open-loop action error on hazardous segments does not correlate with closed-loop success (Spearman {{s:ol_cl_spearman_mae_hazard:2}}).
The gain of CARE is significant only at 1-2% budgets and does not replicate in an indoor mobile-robot domain or in open-loop evaluation on real driving video. The risk-bias collapse and its trade-off, by contrast, appear in all three settings.

The scorer has {{s:cost_scorer_params:0}} parameters and runs in {{s:cost_scorer_onnx_ms:2}} ms on a CPU, which is below 1% of the detector cost. Collection devices should therefore not keep only risky data. They should preserve the nominal distribution and add a small share of risky and uncertain clips chosen by a lightweight context scorer.
