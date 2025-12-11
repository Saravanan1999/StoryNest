## StoryNest Runtime Pipeline (single-turn)

image.png

User Input
     ↓
`SafetyGuard` Agent  
  - Filters / rewrites request to be safe and age-appropriate (5–10).
     ↓
`Story Planner` Agent  
  - Produces structured outline (title, characters, setting, arc, moral).
     ↓
`Story Generator` Agent  
  - Generates full bedtime story from the plan.
     ↓
`Judge` Agent  
  - Scores story quality & safety.
     ↓
(score < threshold?) ── Yes ─────→ `Refinement` Agent ─────→ back to `Judge`  
                    └─ No ───────→ Final Story Output

Final Story Output

## Offline Evaluation & Metrics

Example Prompts (eval set)
     ↓
StoryNest Runtime Pipeline (above)
     ↓
`Judge` Agent score (0–10)  
`StoryEvalMetrics` (helpfulness, safety, coherence, verbosity, relevance, goal_completion, conversation_score, final_score)
     ↓
JSONL Writer (`evals/storynest_results.jsonl`)
     ↓
Logs & Analysis