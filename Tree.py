import enum
import math
import sys

INITIAL_DEPTH = 1

def entropy(data_set):
    entropy = 0
    class_column = data_set.columns[-1]
    frequency_dict = data_set[class_column].value_counts().to_dict()
    for class_value, occur in frequency_dict.items():
        entropy += (occur/len(data_set.index)) * math.log((occur/len(data_set.index)), 2)
    return -1 * entropy


def frequency(data_set, attribute, value):
    df = data_set[(data_set[attribute] == value)]
    return len(df.index) / len(data_set.index)


class NodeType(enum.Enum):
    attribute = 0
    value = 1
    classification = 2


# Clase nodo donde se define cada uno de los nodos que voy a tener. Un nodo puede ser  nodeType:(atributo o valor) y
# value: (Clase, etc. o 1,2,3, etc.), para cada atributo le van a corresponder sus respectivos valores
# (estos se sacan a través de otra función)
class Node:

    def __init__(self, node_type, value, data=None):
        self.nodeType = node_type
        self.value = value
        self.children = []  # Son los hijos del nodo en concreto con el que estoy trabajando
        # para los nodos clasificacion
        self.data = data

    # Función que agrega un hijo al nodo actual siempre y cuando los mismos no tengan igual nodeType y
    # sabiendo que un valor
    def add_child(self, child):
        # no puede tener más de un hijo.
        if self.nodeType == child.nodeType or (self.nodeType == NodeType.value and len(self.children) > 0):
            print(child.value + "Flasheaste")
        else:
            self.children.append(child)


# clase que me genera un arbol el cual parte una raíz
class Tree:

    INITIAL_DEPTH = 1

    def __init__(self):
        self.variables = None
        # diccionario con todos los attr disponibles y sus valores posible ({attr -> [values] })
        self.attribute_dictionary = None
        self.training_set = None
        self.root = None
        self.class_column = None
        self.max_test_depth = None
        self.max_build_depth = None
        self.tree_depth = 0
        self.min_elements_for_fork = None
        self.max_nodes = None
        self.nodes = 0
        self.leaves = []

    def get_class_column(self):
        return self.class_column

    def get_node_amount(self):
        return self.nodes

    def get_tree_depth(self):
        return self.tree_depth

    def build_attr_dict(self):
        dic = {attribute: self.training_set[attribute].unique() for attribute in self.training_set.columns}
        del dic[self.class_column]
        return dic

    def train(self, data_set, max_depth=sys.maxsize, min_elements_for_fork=1, max_nodes=sys.maxsize):
        self.max_build_depth = max_depth
        self.min_elements_for_fork = min_elements_for_fork
        self.training_set = data_set
        self.class_column = data_set.columns[-1]
        self.attribute_dictionary = self.build_attr_dict()
        self.max_nodes = max_nodes
        self.root = None
        self.root = self.build_tree(self.root, self.training_set, INITIAL_DEPTH)

    # {elem-index -> class} y test_elem[-1] como clase real por otro
    def test(self, test_set, depth=sys.maxsize):
        self.max_test_depth = depth
        classifications = {}
        for i in range(len(test_set)):
            test_element = test_set.iloc[i, :]
            classifications[i] = self.traverse_tree(self.root, test_element, INITIAL_DEPTH)

        return classifications

    def traverse_tree(self, curr_node, element, current_depth):
        if curr_node.nodeType == NodeType.classification:
            return curr_node.value
        if curr_node.nodeType == NodeType.attribute:
            element_value = element[curr_node.value]
            for child in curr_node.children:
                if child.value == element_value:
                    return self.traverse_tree(child, element, current_depth+1)
            return None
        else:
            # nodo de tipo value, solo puede tener como hijo a un siguiente attr
            #if current_depth > self.max_test_depth - 1:
            #    self.can_classify()
            return self.traverse_tree(curr_node.children[0], element, current_depth+1)

    def get_next_attribute(self, data_set):
        max_gain = 0
        next_attr = None
        for attr in self.attribute_dictionary.keys():
            gain = self.entropy_gain(data_set, attr)
            if gain > max_gain:
                max_gain = gain
                next_attr = attr
        return next_attr

    def entropy_gain(self, data_set, attr):
        attr_entropy = 0
        for value in self.attribute_dictionary[attr]:
            attr_entropy += frequency(data_set, attr, value) * entropy(data_set[(data_set[attr] == value)])
        return entropy(data_set) - attr_entropy

    def build_tree(self, curr_node, data_set, current_depth):
        if current_depth > self.tree_depth:
            self.tree_depth = current_depth
        if curr_node is None:
            # vengo de node attr
            next_attr = self.get_next_attribute(data_set)
            if next_attr is not None:
                curr_node = self.build_attribute_sub_tree(next_attr)
                # este atributo ya fue usado
                attr_values = self.attribute_dictionary[next_attr]
                del self.attribute_dictionary[next_attr]
                # curr_node es un nodo atributo con sus values como nodos hijos
                for node in curr_node.children:
                    self.build_tree(node, data_set[(data_set[next_attr] == node.value)], current_depth+1)
                self.attribute_dictionary[next_attr] = attr_values
            else:
                # ya use todos los atributos
                classified, class_value, frequency_dict = self.can_classify(data_set, force_classification=True)
                leaf = Node(NodeType.classification, class_value, data=frequency_dict)
                if class_value is not None:
                    self.nodes += 1
                    self.leaves.append(leaf)
                return leaf
        else:
            # estoy en un nodo value
            # en el checkeo de profundidad tengo que tener en cuenta que si no fuerzo aca, voy a tener
            # un nodo atributo y sus nodos valores (2 niveles de profundidad) mas
            force_classification = (self.max_build_depth - 2 <= current_depth or len(data_set.index) < self.min_elements_for_fork or self.nodes+1 >= self.max_nodes)
            classified, class_value, frequency_dict = self.can_classify(data_set, force_classification)
            if not classified:
                curr_node.add_child(self.build_tree(None, data_set, current_depth+1))
            else:
                leaf = Node(NodeType.classification, class_value, data=frequency_dict)
                if class_value is not None:
                    self.leaves.append(leaf)
                    self.nodes += 1
                curr_node.add_child(leaf)
        return curr_node

    def build_attribute_sub_tree(self, attribute):
        attribute_root = Node(NodeType.attribute, attribute)
        self.nodes += 1
        for value in self.attribute_dictionary[attribute]:
            attribute_root.add_child(Node(NodeType.value, value))
            self.nodes += 1
        return attribute_root

    # devuelve si pudo clasificar los datos y su clasificacion en caso de poder. None sino
    def can_classify(self, data_set, force_classification=False):
        frequency_dict = data_set[self.class_column].value_counts().to_dict()
        if len(frequency_dict) == 1:
            # todos los ejemplos son de la misma clase, estoy en condiciones de clasificar
            return True, list(frequency_dict.keys())[0], frequency_dict  # la unica clase
        elif force_classification:
            # si por poda necesito devolver una clasificacion si o si
            max_v = 0
            classification = None
            for k, v in frequency_dict.items():
                if max_v < v:
                    classification = k
                    max_v = v
            return True, classification, frequency_dict
        else:
            # no clasifique, entonces no devuelvo nada
            return False, None, None

