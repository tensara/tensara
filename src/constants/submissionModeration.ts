export const SUBMISSION_MODERATION_LABELS = {
  SUSPICIOUS: "Under Review",
  INVALIDATED: "Invalidated",
} as const;

export const SUBMISSION_MODERATION_TOOLTIPS = {
  SUSPICIOUS:
    "This submission was flagged as suspicious. It is temporarily hidden from leaderboards while the developers review it.",
  INVALIDATED:
    "This submission was invalidated and no longer counts toward public or competitive views, including leaderboards.",
} as const;
