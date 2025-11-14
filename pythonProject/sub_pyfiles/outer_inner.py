class  Person:
    count = 0

    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        # print('args=', cls, args)
        # print('kwargs=', cls, kwargs)
        cls.count += 1
        return instance

    def __init__(self, *args, **kwargs):
        params = list(args[:3])
        if params:
            while len(params) < 3:
                params.append('未知')
            self.name, self.sex, self.age = params
        self.kwargs = kwargs
        for i in self.kwargs.keys():
            self.__setattr__(i, kwargs[i])


    def __str__(self):
        return 'hello' + self.name + 'hello'

    def __call__(self, *args, **kwargs):
        return f'{self.name}\'s age is {self.age}'

    def __del__(self):
        self.__class__.count -= 1
        print('del', self.__class__, self)

    def say_hello(self):
        print(f"hello,I'm {self.name}!")


obj1 = Person('张三', '男', '18')
print(obj1)
obj1.say_hello()


