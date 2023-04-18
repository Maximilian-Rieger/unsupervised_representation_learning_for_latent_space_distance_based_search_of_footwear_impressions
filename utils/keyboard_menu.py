from enum import Enum
import logging

class ArgumentType(Enum):
    LIST_INT_FLOAT = 4


class MenuItem:
    def __init__(self, description, method=None, arg_type=None):
        self.description = description
        self.method = method
        self.arg_type = arg_type

    def __call__(self, *args, **kwargs):
        if self.method is not None:
            self.method(*args)


class KeyboardMenu:
    def __init__(self, abort_method, items=None):
        self.menu_items = [MenuItem("continue"),
                           MenuItem("abort", abort_method)]

        if items:
            for i in items:
                self.add_item(**i)

    def add_item(self, description, method, arg_type=None):
        item = MenuItem(description=description,
                        method=method,
                        arg_type=arg_type,)
        self.menu_items.insert(len(self.menu_items) - 1, item)
        return self

    def print_menu(self):
        print('\n')
        for i, m in enumerate(self.menu_items):
            print("(%s)\t%s" % (i, m.description))

    def show_menu(self):
        self.print_menu()
        new_input = -1
        while new_input < 0 or new_input > len(self.menu_items):
            try:
                new_input = int(input("Your x:"))
            except ValueError:
                new_input = -1

        if self.menu_items[new_input].arg_type:
            valid_input = False
            arg = None

            while not valid_input:
                try:
                    arg_input = input('value: ')

                    if self.menu_items[new_input].arg_type == int:
                        arg = int(arg_input)
                        valid_input = True
                    elif self.menu_items[new_input].arg_type == float:
                        arg = float(arg_input)
                        valid_input = True
                    elif self.menu_items[new_input].arg_type == ArgumentType.LIST_INT_FLOAT:
                        import re

                        list_string = re.search('\[(.*)\]', arg_input).group(1)
                        tuples_iter = re.finditer('\((\d+),\s*(\d+\.?\d*)\)', list_string)

                        arg = []
                        for t in tuples_iter:
                            epoch = int(t.group(1))
                            lr = float(t.group(2))
                            arg.append((epoch, lr))
                        if len(arg) > 0:
                            print("invalid x")
                            valid_input = True
                    else:
                        logging.warning("argument type not knwon")

                except ValueError:
                    print("invalid x")

            self.menu_items[new_input](arg)
        else:
            self.menu_items[new_input]()

    def __call__(self):
        if self.heard_enter():
            self.show_menu()

    def heard_enter(self):
        """
        Check if the used has interrupted
        :return: if the user has interrupted
        :rtype: bool
        """
        import select
        import sys
        i, o, e = select.select([sys.stdin], [], [], 0.0001)
        for s in i:
            if s == sys.stdin:
                # x = sys.stdin.readline()
                sys.stdin.readline()  # consume enter
                return True
        return False
