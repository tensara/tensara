---
title: "The Tensara Score"
date: "2025-03-17"
---

The Tensara scoring system is designed to provide a comprehensive measure of user performance on our platform. Unlike simple metrics that only count submissions or track raw performance, our enhanced scoring system takes into account multiple factors to reward skill, innovation, efficiency, and consistency.

## Core Components

The Tensara score is calculated based on five key components:

$$
\text{Score} = \sum_{p \in \text{Problems}} \left( \text{BaseScore}_p \times \text{Factors} \right) \times \text{ConsistencyBonus}
$$

Where the factors include:

1. **Problem Difficulty Multiplier**
2. **Improvement Bonus**
3. **First-Solver Bonus**
4. **Diminishing Returns for Repeated Solutions**
5. **Consistency Bonus**

## Problem Difficulty Multiplier

Problems are categorized by difficulty, with higher multipliers for more challenging problems:

| Difficulty | Multiplier |
|------------|------------|
| EASY       | 1.0x       |
| MEDIUM     | 1.5x       |
| HARD       | 2.5x       |
| EXTREME    | 4.0x       |

The base score (GFLOPS achieved) is multiplied by this difficulty factor:

$$
\text{BaseScore}_p = \text{GFLOPS} \times \text{DifficultyMultiplier}
$$

This ensures that solving harder problems contributes significantly more to your overall score than easier ones, even if the raw GFLOPS are similar.

## Improvement Bonus

If you submit an improved solution to a problem you've already solved, you receive a bonus proportional to the percentage improvement:

$$
\text{ImprovementBonus} = \frac{\text{NewGFLOPS} - \text{PreviousBestGFLOPS}}{\text{PreviousBestGFLOPS}} \times 0.5
$$

The improvement bonus is capped at 50% (for a solution that's at least twice as good as your previous best), and is applied as:

$$
\text{Score}_{\text{improved}} = \text{BaseScore} \times (1 + \text{ImprovementBonus})
$$

This rewards continued optimization while preventing excessive scores from minor tweaks to existing solutions.

## First-Solver Bonus

Being among the first 5 users to solve a problem earns a significant bonus:

$$
\text{FirstSolveMultiplier} = 1 + \frac{\text{Bonus} \times (\text{Count} - \text{Position})}{\text{Count}}
$$

Where:
- Bonus = 10
- Count = 5
- Position = Your solving position (0-indexed)

For example:
- First solver: 1 + (10 × 5/5) = 11x multiplier
- Second solver: 1 + (10 × 4/5) = 9x multiplier
- Third solver: 1 + (10 × 3/5) = 7x multiplier

This rewards users who tackle and solve new problems quickly, encouraging early adoption of newly released challenges.

## Diminishing Returns

The diminishing returns mechanism ensures that users focus on quality improvements rather than making many small, incremental submissions:

$$
\text{Score}_{\text{considered}} = \max_{s \in \text{Submissions}_p} \text{Score}(s)
$$

Where:
- $\text{Submissions}_p$ is the set of all user submissions for problem $p$
- $\text{Score}(s)$ is the calculated score for submission $s$

For each problem, we track the percentage improvement over your previous best:

$$
\text{Improvement}(s_{\text{new}}, s_{\text{best}}) = \frac{\text{GFLOPS}(s_{\text{new}}) - \text{GFLOPS}(s_{\text{best}})}{\text{GFLOPS}(s_{\text{best}})}
$$

A new submission only contributes to your score if it shows a meaningful improvement. The system implements a threshold function:

$$
\text{IsSignificant}(s_{\text{new}}, s_{\text{best}}) = 
\begin{cases}
\text{true}, & \text{if } \text{Improvement}(s_{\text{new}}, s_{\text{best}}) \geq 0.05 \\
\text{false}, & \text{otherwise}
\end{cases}
$$

This means that only submissions with at least a 5% performance improvement over your previous best are considered for scoring. This prevents users from gaining inflated scores by repeatedly submitting similar solutions with minimal improvements.

## Consistency Bonus

Regular activity on the platform over a 30-day window is rewarded with a consistency bonus:

$$
\text{ConsistencyFactor} = \min(1, \frac{\text{UniqueSubmissionDays}}{\text{WindowDays}})
$$

$$
\text{ConsistencyBonus} = 1 + (0.2 \times \text{ConsistencyFactor})
$$

The consistency bonus is applied to your total score and can provide up to a 20% boost for users who submit solutions on many different days throughout the month.

## Example Calculation

Consider a user who:
1. Solved an EXTREME problem with 100 GFLOPS (first solver)
2. Solved a HARD problem with 150 GFLOPS (third solver)
3. Improved their solution to the EXTREME problem to 150 GFLOPS (50% improvement)
4. Has been active on 15 out of the last 30 days

Their score would be calculated as:

$$
\begin{align*}
\text{Score}_{\text{first EXTREME}} &: \text{Not counted (superseded by improved solution)} \\
\text{Score}_{\text{improved EXTREME}} &= 150 \times 4.0 \times (1 + (0.5 \times 0.5)) \times 11 = 9900 \\
\text{Score}_{\text{HARD}} &= 150 \times 2.5 \times 7 = 2625 \\
\text{ConsistencyBonus} &= 1 + \left(0.2 \times \frac{15}{30}\right) = 1.1 \\
\text{TotalScore} &= (9900 + 2625) \times 1.1 = 13777.5
\end{align*}
$$

This comprehensive scoring system ensures that users are rewarded not just for raw computational performance, but for tackling difficult problems, being early solvers, continuously improving their solutions, and maintaining consistent activity on the platform. 