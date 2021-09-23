import enum
import math


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

    def __init__(self, node_type, value):
        self.nodeType = node_type
        self.value = value
        self.children = []  # Son los hijos del nodo en concreto con el que estoy trabajando

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
    def __init__(self):
        self.variables = None
        self.filters = {}
        # diccionario con todos los attr disponibles y sus valores posible ({attr -> [values] })
        self.attribute_dictionary = None
        self.training_set = None
        self.root = None
        self.class_column = None

    def get_class_column(self):
        return self.class_column

    def build_attr_dict(self):
        dict = {attribute: self.training_set[attribute].unique() for attribute in self.training_set.columns}
        del dict[self.class_column]
        return dict

    def train(self, data_set):
        self.training_set = data_set
        self.class_column = data_set.columns[-1]
        self.attribute_dictionary = self.build_attr_dict()
        self.root = self.build_tree(self.root, self.training_set)

    # {elem-index -> class} y test_elem[-1] como clase real por otro
    def test(self, test_set):
        classifications = {}
        for i in range(len(test_set)):
            test_element = test_set.iloc[i, :]
            classifications[i] = self.traverse_tree(self.root, test_element)

        return classifications

    def traverse_tree(self, curr_node, element):
        if curr_node.nodeType == NodeType.classification:
            return curr_node.value
        if curr_node.nodeType == NodeType.attribute:
            element_value = element[curr_node.value]
            for child in curr_node.children:
                if child.value == element_value:
                    return self.traverse_tree(child, element)
            raise Exception("training set does not have value ", element_value, " for attribute ", curr_node.value)
        else:
            # nodo de tipo value, solo puede tener como hijo a un siguiente attr
            return self.traverse_tree(curr_node.children[0], element)

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

    def build_tree(self, curr_node, data_set):
        if curr_node is None:
            # vengo de node attr
            next_attr = self.get_next_attribute(data_set)
            if next_attr is not None:
                curr_node = self.build_attribute_sub_tree(next_attr)
                # este atributo ya fue usado
                del self.attribute_dictionary[next_attr]
                # curr_node es un nodo atributo con sus values como nodos hijos
                for node in curr_node.children:
                    self.build_tree(node, data_set[(data_set[next_attr] == node.value)])
            else:
                # ya use todos los atributos
                classified, class_value = self.can_classify(data_set, force_classification=True)
                return Node(NodeType.classification, class_value)
        else:
            # estoy en un nodo value
            classified, class_value = self.can_classify(data_set)
            if not classified:
                curr_node.add_child(self.build_tree(None, data_set))
            else:
                curr_node.add_child(Node(NodeType.classification, class_value))

        return curr_node

    def build_attribute_sub_tree(self, attribute):
        attribute_root = Node(NodeType.attribute, attribute)
        for value in self.attribute_dictionary[attribute]:
            attribute_root.add_child(Node(NodeType.value, value))
        return attribute_root

    # devuelve si pudo clasificar los datos y su clasificacion en caso de poder. None sino
    def can_classify(self, data_set, force_classification=False):
        class_column = data_set.columns[-1]
        frequency_dict = data_set[class_column].value_counts().to_dict()
        if len(frequency_dict) == 1:
            # todos los ejemplos son de la misma clase, estoy en condiciones de clasificar
            return True, list(frequency_dict.keys())[0]  # la unica clase
        elif force_classification:
            # si por poda necesito devolver una clasificacion si o si
            max_v = 0
            classification = None
            for k, v in frequency_dict.items():
                if max_v < v:
                    classification = k
                    max_v = v
            return True, classification
        else:
            # no clasifique, entonces no devuelvo nada
            return False, None

