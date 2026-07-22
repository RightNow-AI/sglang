# Goal

AutoTree makes tree-structured reasoning a first-class serving primitive.

Language models reason better when they can explore several lines of thought and
keep the best one. Today's serving engines run each line as an isolated request.
AutoTree runs them as a tree: fork a generation at token granularity, explore
branches that share their common history, prune the weak ones, and return the
branch that wins - all behind an OpenAI-compatible API, as a drop-in extension
of SGLang.

The aim is simple: make branching, voting, and pruning over reasoning as easy
and efficient to serve as a single completion is today.
