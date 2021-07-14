from sys import setrecursionlimit

setrecursionlimit(10 ** 6)


class BSTNode(object):
    """A node in the vanilla BST tree."""

    def __init__(self, parent, k):
        """Creates a node.

        Args:
            parent: The node's parent.
            k: key of the node.
        """
        self.key = k[0]
        self.parent = parent
        self.left = None
        self.right = None
        self.real = k

    def _str(self):
        """Internal method for ASCII art."""
        label = str(self.key)
        if self.left is None:
            left_lines, left_pos, left_width = [], 0, 0
        else:
            left_lines, left_pos, left_width = self.left._str()
        if self.right is None:
            right_lines, right_pos, right_width = [], 0, 0
        else:
            right_lines, right_pos, right_width = self.right._str()
        middle = max(right_pos + left_width - left_pos + 1, len(label), 2)
        pos = left_pos + middle // 2
        width = left_pos + middle + right_width - right_pos
        while len(left_lines) < len(right_lines):
            left_lines.append(' ' * left_width)
        while len(right_lines) < len(left_lines):
            right_lines.append(' ' * right_width)
        if (middle - len(label)) % 2 == 1 and self.parent is not None and \
                self is self.parent.left and len(label) < middle:
            label += '.'
        label = label.center(middle, '.')
        if label[0] == '.': label = ' ' + label[1:]
        if label[-1] == '.': label = label[:-1] + ' '
        lines = [' ' * left_pos + label + ' ' * (right_width - right_pos),
                 ' ' * left_pos + '/' + ' ' * (middle - 2) +
                 '\\' + ' ' * (right_width - right_pos)] + \
                [left_line + ' ' * (width - left_width - right_width) + right_line
                 for left_line, right_line in zip(left_lines, right_lines)]
        return lines, pos, width

    def __str__(self):
        return '\n'.join(self._str()[0])

    def find(self, k):
        """Finds and returns the node with key k from the subtree rooted at this
        node.

        Args:
            k: The key of the node we want to find.

        Returns:
            The node with key k.
        """
        if k[0] == self.key:
            return self
        elif k[0] < self.key:
            if self.left is None:
                return None
            else:
                return self.left.find(k)
        else:
            if self.right is None:
                return None
            else:
                return self.right.find(k)

    def find_min(self):
        """Finds the node with the minimum key in the subtree rooted at this
        node.

        Returns:
            The node with the minimum key.
        """
        current = self
        while current.left is not None:
            current = current.left
        return current

    def next_larger(self):
        """Returns the node with the next larger key (the successor) in the BST.
        """
        if self.right is not None:
            return self.right.find_min()
        current = self
        while current.parent is not None and current is current.parent.right:
            current = current.parent
        return current.parent

    def insert(self, node):
        """Inserts a node into the subtree rooted at this node.

        Args:
            node: The node to be inserted.
        """
        if node is None:
            return
        if node.key < self.key:
            if self.left is None:
                node.parent = self
                self.left = node
            else:
                self.left.insert(node)
        else:
            if self.right is None:
                node.parent = self
                self.right = node
            else:
                self.right.insert(node)

    def delete(self):
        """Deletes and returns this node from the BST."""
        if self.left is None or self.right is None:
            if self is self.parent.left:
                self.parent.left = self.left or self.right
                if self.parent.left is not None:
                    self.parent.left.parent = self.parent
            else:
                self.parent.right = self.left or self.right
                if self.parent.right is not None:
                    self.parent.right.parent = self.parent
            return self
        else:
            s = self.next_larger()
            self.key, s.key = s.key, self.key
            return s.delete()

    def check_ri(self):
        """Checks the BST representation invariant around this node.

        Raises an exception if the RI is violated.
        """
        if self.left is not None:
            if self.left.key > self.key:
                raise RuntimeError("BST RI violated by a left node key")
            if self.left.parent is not self:
                raise RuntimeError("BST RI violated by a left node parent "
                                   "pointer")
            self.left.check_ri()
        if self.right is not None:
            if self.right.key < self.key:
                raise RuntimeError("BST RI violated by a right node key")
            if self.right.parent is not self:
                raise RuntimeError("BST RI violated by a right node parent "
                                   "pointer")
            self.right.check_ri()


class BST(object):
    """A binary search tree."""

    def __init__(self, klass=BSTNode):
        """Creates an empty BST.

        Args:
            klass (optional): The class of the node in the BST. Default to
                BSTNode.
        """
        self.root = None
        self.klass = klass

    def __str__(self):
        if self.root is None: return '<empty tree>'
        return str(self.root)

    def find(self, k):
        """Finds and returns the node with key k from the subtree rooted at this
        node.

        Args:
            k: The key of the node we want to find.

        Returns:
            The node with key k or None if the tree is empty.
        """
        return self.root and self.root.find(k)

    def find_min(self):
        """Returns the minimum node of this BST."""

        return self.root and self.root.find_min()

    def insert(self, k):
        """Inserts a node with key k into the subtree rooted at this node.

        Args:
            k: The key of the node to be inserted.

        Returns:
            The node inserted.
        """
        node = self.klass(None, k)
        if self.root is None:
            # The root's parent is None.
            self.root = node
        else:
            self.root.insert(node)
        return node

    def delete(self, k):
        """Deletes and returns a node with key k if it exists from the BST.

        Args:
            k: The key of the node that we want to delete.

        Returns:
            The deleted node with key k.
        """
        node = self.find(k)
        if node is None:
            return None
        if node is self.root:
            pseudoroot = self.klass(None, (0, None))
            pseudoroot.left = self.root
            self.root.parent = pseudoroot
            deleted = self.root.delete()
            self.root = pseudoroot.left
            if self.root is not None:
                self.root.parent = None
            return deleted
        else:
            return node.delete()

    def next_larger(self, k):
        """Returns the node that contains the next larger (the successor) key in
        the BST in relation to the node with key k.

        Args:
            k: The key of the node of which the successor is to be found.

        Returns:
            The successor node.
        """
        node = self.find(k)
        return node and node.next_larger()

    def check_ri(self):
        """Checks the BST representation invariant.

        Raises:
            An exception if the RI is violated.
        """
        if self.root is not None:
            if self.root.parent is not None:
                raise RuntimeError("BST RI violated by the root node's parent "
                                   "pointer.")
            self.root.check_ri()


def height(node):
    if node is None:
        return -1
    else:
        return node.height


def update_height(node):
    node.height = max(height(node.left), height(node.right)) + 1


class AVL(BST):
    """
AVL binary search tree implementation.
Supports insert, find, and delete-min operations in O(lg n) time.
"""

    def left_rotate(self, x):
        y = x.right
        y.parent = x.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            elif y.parent.right is x:
                y.parent.right = y
        x.right = y.left
        if x.right is not None:
            x.right.parent = x
        y.left = x
        x.parent = y
        update_height(x)
        update_height(y)

    def right_rotate(self, x):
        y = x.left
        y.parent = x.parent
        if y.parent is None:
            self.root = y
        else:
            if y.parent.left is x:
                y.parent.left = y
            elif y.parent.right is x:
                y.parent.right = y
        x.left = y.right
        if x.left is not None:
            x.left.parent = x
        y.right = x
        x.parent = y
        update_height(x)
        update_height(y)

    def insert(self, t):
        """Insert key t into this tree, modifying it in-place."""
        node = BST.insert(self, t)
        self.rebalance(node)

    def rebalance(self, node):
        while node is not None:
            update_height(node)
            if height(node.left) >= 2 + height(node.right):
                if height(node.left.left) >= height(node.left.right):
                    self.right_rotate(node)
                else:
                    self.left_rotate(node.left)
                    self.right_rotate(node)
            elif height(node.right) >= 2 + height(node.left):
                if height(node.right.right) >= height(node.right.left):
                    self.left_rotate(node)
                else:
                    self.right_rotate(node.right)
                    self.left_rotate(node)
            node = node.parent

    def delete_min(self):
        node, parent = delete_min(self)
        self.rebalance(parent)
        # raise NotImplemented('AVL.delete_min')

    def List(self, l, h):
        def LCA(self, l, h):
            node = self.root
            while ((node == None or (l <= node.key and node.key <= h)) == False):
                if (l < node.key):
                    node = node.left
                else:
                    node = node.right
            return node

        def NODELIST(node, l, h, result):
            if (node == None):
                return []
            if l <= node.key and node.key <= h:
                result.append(node)
            if (node.key >= l):
                NODELIST(node.left, l, h, result)
            if (node.key <= h):
                NODELIST(node.right, l, h, result)

        lca = LCA(self, l, h)
        result = []
        NODELIST(lca, l, h, result)
        return result


import json  # Used when TRACE=jsonp
import os  # Used to get the TRACE environment variable
import re  # Used when TRACE=jsonp
import sys  # Used to smooth over the range / xrange issue.

# Python 3 doesn't have xrange, and range behaves like xrange.
if sys.version_info >= (3,):
    xrange = range


# Circuit verification library.

class Wire(object):
    """A wire in an on-chip circuit.

    Wires are immutable, and are either horizontal or vertical.
    """

    def __init__(self, name, x1, y1, x2, y2):
        """Creates a wire.

        Raises an ValueError if the coordinates don't make up a horizontal wire
        or a vertical wire.

        Args:
          name: the wire's user-visible name
          x1: the X coordinate of the wire's first endpoint
          y1: the Y coordinate of the wire's first endpoint
          x2: the X coordinate of the wire's last endpoint
          y2: the Y coordinate of the wire's last endpoint
        """
        # Normalize the coordinates.
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        self.name = name
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.object_id = Wire.next_object_id()

        if not (self.is_horizontal() or self.is_vertical()):
            raise ValueError(str(self) + ' is neither horizontal nor vertical')

    def is_horizontal(self):
        """True if the wire's endpoints have the same Y coordinates."""
        return self.y1 == self.y2

    def is_vertical(self):
        """True if the wire's endpoints have the same X coordinates."""
        return self.x1 == self.x2

    def intersects(self, other_wire):
        """True if this wire intersects another wire."""
        # NOTE: we assume that wires can only cross, but not overlap.
        if self.is_horizontal() == other_wire.is_horizontal():
            return False

        if self.is_horizontal():
            h = self
            v = other_wire
        else:
            h = other_wire
            v = self
        return v.y1 <= h.y1 and h.y1 <= v.y2 and h.x1 <= v.x1 and v.x1 <= h.x2

    def __repr__(self):
        # :nodoc: nicer formatting to help with debugging
        return ('<wire ' + self.name + ' (' + str(self.x1) + ',' + str(self.y1) + ')-(' + str(self.x2) + ',' + str(
            self.y2) + ')>')

    def as_json(self):
        """Dict that obeys the JSON format restrictions, representing the wire."""
        return {'id': self.name, 'x': [self.x1, self.x2], 'y': [self.y1, self.y2]}

    # Next number handed out by Wire.next_object_id()
    _next_id = 0

    @staticmethod
    def next_object_id():
        """Returns a unique numerical ID to be used as a Wire's object_id."""
        id = Wire._next_id
        Wire._next_id += 1
        return id


class WireLayer(object):
    """The layout of one layer of wires in a chip."""

    def __init__(self):
        """Creates a layer layout with no wires."""
        self.wires = {}

    def wires(self):
        """The wires in the layout."""
        self.wires.values()

    def add_wire(self, name, x1, y1, x2, y2):
        """Adds a wire to a layer layout.

        Args:
          name: the wire's unique name
          x1: the X coordinate of the wire's first endpoint
          y1: the Y coordinate of the wire's first endpoint
          x2: the X coordinate of the wire's last endpoint
          y2: the Y coordinate of the wire's last endpoint

        Raises an exception if the wire isn't perfectly horizontal (y1 = y2) or
        perfectly vertical (x1 = x2)."""
        if name in self.wires:
            raise ValueError('Wire name ' + name + ' not unique')
        self.wires[name] = Wire(name, x1, y1, x2, y2)

    def as_json(self):
        """Dict that obeys the JSON format restrictions, representing the layout."""
        return {'wires': [wire.as_json() for wire in self.wires.values()]}

    @staticmethod
    def from_file(file):
        """Builds a wire layer layout by reading a textual description from a file.

        Args:
          file: a File object supplying the input

        Returns a new Simulation instance."""

        layer = WireLayer()
        file = open(str(file), 'r')
        whole = file.readlines()
        for command in whole:
            command = command.split()
            if command[0] == 'wire':
                coordinates = [float(token) for token in command[2:6]]
                layer.add_wire(command[1], *coordinates)
            elif command[0] == 'done':
                break
        return layer


class RangeIndex(object):
    def __init__(self):
        self.data = AVL()

    def add(self, n):
        self.data.insert(n)

    def remove(self, n):
        self.data.delete(n)

    def list(self, first_key, last_key):
        return self.data.List(first_key, last_key)

    def count(self, first_key, last_key):
        return len(self.data.List(first_key, last_key))


class ResultSet(object):
    """Records the result of the circuit verifier (pairs of crossing wires)."""

    def __init__(self):
        """Creates an empty result set."""
        self.crossings = []

    def add_crossing(self, wire1, wire2):
        """Records the fact that two wires are crossing."""
        self.crossings.append(sorted([str(wire1.name), str(wire2.name)]))

    def write_to_file(self, file):
        """Write the result to a file."""
        for crossing in self.crossings:
            file.write(' '.join(crossing))
            file.write('\n')


class KeyWirePair(object):
    """Wraps a wire and the key representing it in the range index.

    Once created, a key-wire pair is immutable."""

    def __init__(self, key, wire):
        """Creates a new key for insertion in the range index."""
        self.key = key
        if wire is None:
            raise ValueError('Use KeyWirePairL or KeyWirePairH for queries')
        self.wire = wire
        self.wire_id = wire.object_id

    def __lt__(self, other):
        # :nodoc: Delegate comparison to keys.
        return (self.key < other.key or
                (self.key == other.key and self.wire_id < other.wire_id))

    def __le__(self, other):
        # :nodoc: Delegate comparison to keys.
        return (self.key < other.key or
                (self.key == other.key and self.wire_id <= other.wire_id))

    def __gt__(self, other):
        # :nodoc: Delegate comparison to keys.
        return (self.key > other.key or
                (self.key == other.key and self.wire_id > other.wire_id))

    def __ge__(self, other):
        # :nodoc: Delegate comparison to keys.
        return (self.key > other.key or (self.key == other.key and self.wire_id >= other.wire_id))

    def __eq__(self, other):
        # :nodoc: Delegate comparison to keys.
        return self.key == other.key and self.wire_id == other.wire_id

    def __ne__(self, other):
        # :nodoc: Delegate comparison to keys.
        return self.key == other.key and self.wire_id == other.wire_id

    def __hash__(self):
        # :nodoc: Delegate comparison to keys.
        return hash([self.key, self.wire_id])

    def __repr__(self):
        # :nodoc: nicer formatting to help with debugging
        return '<key: ' + str(self.key) + ' wire: ' + str(self.wire) + '>'


class KeyWirePairL(KeyWirePair):
    """A KeyWirePair that is used as the low end of a range query.

    This KeyWirePair is smaller than all other KeyWirePairs with the same key."""

    def __init__(self, key):
        self.key = key
        self.wire = None
        self.wire_id = -1000000000


class KeyWirePairH(KeyWirePair):
    """A KeyWirePair that is used as the high end of a range query.

    This KeyWirePair is larger than all other KeyWirePairs with the same key."""

    def __init__(self, key):
        self.key = key
        self.wire = None
        # HACK(pwnall): assuming 1 billion objects won't fit into RAM.
        self.wire_id = 1000000000


class CrossVerifier(object):
    """Checks whether a wire network has any crossing wires."""

    def __init__(self, layer):
        """Verifier for a layer of wires.

        Once created, the verifier can list the crossings between wires (the
        wire_crossings method) or count the crossings (count_crossings)."""

        self.events = []
        self._events_from_layer(layer)
        self.events.sort()

        self.index = RangeIndex()
        self.result_set = ResultSet()
        self.performed = False

    def count_crossings(self):
        """Returns the number of pairs of wires that cross each other."""
        if self.performed:
            raise
        self.performed = True
        return self._compute_crossings(True)

    def wire_crossings(self):
        """An array of pairs of wires that cross each other."""
        if self.performed:
            raise
        self.performed = True
        return self._compute_crossings(False)

    def _events_from_layer(self, layer):
        """Populates the sweep line events from the wire layer."""
        # left_edge = min([wire.x1 for wire in layer.wires.values()])
        for wire in layer.wires.values():
            if wire.is_horizontal():
                self.events.append([wire.x1, 0, wire.object_id, 'add', wire])
                self.events.append([wire.x2, 2, wire.object_id, "remove", wire])
            else:
                self.events.append([wire.x1, 1, wire.object_id, 'query', wire])

    def _compute_crossings(self, count_only):
        """Implements count_crossings and wire_crossings."""
        if count_only:
            result = 0
        else:
            result = self.result_set

        for event in self.events:
            event_x, event_type, wire = event[0], event[3], event[4]

            if event_type == 'add':
                self.index.add((wire.y1, wire))
            elif (event_type == "remove"):
                self.index.remove((wire.y1, wire))
            elif event_type == 'query':
                Y1 = wire.y1
                Y2 = wire.y2
                if (Y1 > Y2):
                    Y1, Y2 = Y2, Y1
                cross_wires = self.index.list((Y1), (Y2))
                if count_only:
                    result += len(cross_wires)
                else:
                    for cross_wire in cross_wires:
                        result.add_crossing(wire, cross_wire.real[1])
        return result


# Command-line controller.
if __name__ == '__main__':
    import sys

    layer = WireLayer.from_file("10grid_s.in")
    verifier = CrossVerifier(layer)

    if os.environ.get('TRACE') == 'jsonp':
        verifier = TracedCrossVerifier(layer)
        result = verifier.wire_crossings()
        json_obj = {'layer': layer.as_json(), 'trace': verifier.trace_as_json()}
        sys.stdout.write('onJsonp(')
        json.dump(json_obj, sys.stdout)
        sys.stdout.write(');\n')
    elif os.environ.get('TRACE') == 'list':
        ## make this elif as True and remove this condition. It will let you write the names of pairs of wires that cross each other
        f=open("10thoutput.txt",'w')
        verifier.wire_crossings().write_to_file(f)
    else:
        f = open("10thoutput.txt", 'w')
        f.write(str(verifier.count_crossings()) + "\n")

