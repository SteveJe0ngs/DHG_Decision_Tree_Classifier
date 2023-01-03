from decisionTree import clf
from sklearn import tree
import graphviz

dot_data = tree.export_graphviz(
    clf,
    out_file=None,
    feature_names=['E/I', 'N/S', 'F/T', 'J/P'],
    class_names=['Divergers', 'Assimilators', 'Convergers', 'Accommodators'],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.render(filename='decision_tree_classifier', directory='./', format='png')