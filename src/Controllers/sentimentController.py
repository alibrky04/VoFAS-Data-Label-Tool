class sentimentController:
    """
    A controller responsible for handling sentiment analysis results from multiple AI models.
    It checks the sentiment predictions from the models and identifies if there are discrepancies 
    in their results. If the sentiment values differ, it flags the feedback for review by a human.

    The controller ensures that feedback is correctly categorized by checking the consistency of 
    model predictions. If models disagree on the sentiment, the feedback will be flagged for manual review.
    """
    def __init__(self):
        """
        Initializes the sentimentController.
        """
        self.flagged_feedback = []