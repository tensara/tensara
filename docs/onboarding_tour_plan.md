# Tensara User Onboarding Tour Plan

## Goal
Guide a new user through the essential workflow of finding, understanding, solving, and submitting a problem on the Tensara platform, familiarizing them with key features like results review and profile tracking.

## Assumptions
*   The tour starts immediately after the user's first sign-in.
*   A specific "Introductory Problem" is used for the guided walkthrough.
*   The tour uses visual cues (highlighting, tooltips) - specific library TBD.

## Tour Flow Diagram

```mermaid
graph TD
    A[Start: Post Sign-in / Dashboard] --> B{Direct to Problems};
    B --> C[Problems List Page (`/problems`)];
    C --> D{Select 'Introductory Problem'};
    D --> E[Problem Page (`/problems/intro-problem`)];

    subgraph Problem Page Interaction
        E --> F[Highlight: Problem Details (Left Panel)];
        F --> G[Highlight: Workspace (Right Panel)];
        G --> H{Instruct: Enter Sample Solution};
        H --> I[Highlight: Submit Button];
        I --> J{Instruct: Click Submit};
    end

    J --> K[Submission Results View];
    subgraph Results & Profile
        K --> L[Highlight: Status, Perf, Leaderboard, Benchmarking, Tests, Errors, Nav];
        L --> M{Navigate to Profile Page};
        M --> N[Profile Page (`/[username]`)];
        N --> O[Highlight: Header, Stats, Activity Calendar, Recent Submissions];
    end

    subgraph Next Steps
        O --> P{Suggest Next Steps};
        P --> Q[Option 1: Try Problems Button (-> `/problems`)];
        P --> R[Option 2: Learn Basics Button (Placeholder)];
        Q --> S[End Tour];
        R --> S;
    end

    style Q fill:#f9f,stroke:#333,stroke-width:2px
    style R fill:#f9f,stroke:#333,stroke-width:2px
```

## Detailed Steps

1.  **Welcome & Intro (Dashboard):**
    *   Show welcome message.
    *   Briefly explain Tensara's purpose.
    *   Prompt user to start tour.
    *   **Action:** Guide user to click "Start Tour" button -> `/problems`.

2.  **Finding a Problem (`/problems` page):**
    *   Highlight the list of problems.
    *   **Action:** Guide user to click the designated "Introductory Problem".

3.  **Understanding the Problem (`/problems/intro-problem` page - Problem View):**
    *   **Introduce Layout:** Explain split-panel (Problem Details left, Workspace right).
    *   **Focus on Left Panel (`ProblemView`):** Highlight Title, Difficulty, Description, Input Spec, Output Spec, Constraints, Examples. Mention "View My Submissions" button purpose.
    *   **Introduce Right Panel (Workspace):** Highlight Language/GPU selectors, Code Editor (explain starter code), Reset Code button.

4.  **Solving the Problem (Code Editor):**
    *   **Action:** Instruct user to replace starter code with a provided simple, correct solution.

5.  **Submitting the Solution:**
    *   Highlight the `Submit` button.
    *   **Action:** Instruct user to click `Submit`.

6.  **Reviewing Results (`SubmissionResults` View):**
    *   Highlight **Submission Status** (Success/Error) and explain.
    *   Highlight **Performance Metrics** (Execution Time, Score) and explain.
    *   Mention **Leaderboard** context.
    *   Provide link/info about **[Benchmarking Blog Post](/blog/benchmarking)**.
    *   Highlight **Test Cases** section (Pass/Fail breakdown) and explain.
    *   If error, highlight **Error Message** area and explain.
    *   Highlight **Navigation Buttons** ("Back to Problem", "View My Submissions") and explain their context.

7.  **Introducing Your Profile Page (`/[username]` page):**
    *   **Transition:** Prompt user to check profile.
    *   **Navigation:** Guide user to their profile page.
    *   **Highlight Key Areas:** User Header, Stats Box, Activity Calendar (showing today's submission), Recent Submissions list (showing the new submission).
    *   **Purpose:** Explain profile tracks progress and history.

8.  **(Optional) Viewing Submissions (`MySubmissions` View):**
    *   Briefly show where the history of submissions for a *specific problem* is stored (can be accessed via "View My Submissions" button on Problem/Results page).

9.  **Next Steps & Conclusion:**
    *   Congratulate user.
    *   Present two options:
        *   **Option A:** "Try another problem" - Button linking to `/problems`.
        *   **Option B:** "Learn Parallel Programming Basics" - Button (initially non-functional placeholder).
    *   **Action:** End the tour when the user clicks either option or an "End Tour" button.