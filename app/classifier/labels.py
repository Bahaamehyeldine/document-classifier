"""RVL-CDIP class labels, in the dataset's official index order (0-15)."""

RVL_CDIP_CLASSES: tuple[str, ...] = (
    "letter",
    "form",
    "email",
    "handwritten",
    "advertisement",
    "scientific_report",
    "scientific_publication",
    "specification",
    "file_folder",
    "news_article",
    "budget",
    "invoice",
    "presentation",
    "questionnaire",
    "resume",
    "memo",
)

NUM_CLASSES = len(RVL_CDIP_CLASSES)
