export const SUBMISSION_MODERATION_LABELS = {
  SUSPICIOUS: "Under Review",
  INVALIDATED: "Invalidated",
} as const;

export const SUBMISSION_MODERATION_TOOLTIPS = {
  SUSPICIOUS:
    "This submission was flagged as suspicious. It is temporarily hidden from leaderboards while the developers review it.",
  INVALIDATED:
    "This submission was removed by the developers and no longer counts toward public or competitive views. Please reach out to the developers on Discord if you have any questions about this submission.",
} as const;
