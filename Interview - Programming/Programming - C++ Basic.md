
以下是 C++ 基礎的 80 個面試題目，涵蓋了 C++ 語言基礎、面向對象編程、模板、STL、內存管理等核心概念，適合用來準備初中級 C++ 程式設計面試。

### 1. C++ 語言基礎

1. C++ 與 C 語言有什麼區別？
2. 如何使用 namespace？namespace 的作用是什麼？
3. 為什麼要使用 `using namespace std;`？
4. C++ 中的作用域解析運算符 `::` 是什麼？
5. 如何在 C++ 中宣告常量？const 和 constexpr 的區別是什麼？
6. C++ 中的引用和指針有什麼區別？
7. 什麼是右值引用？右值引用的應用場景有哪些？
8. C++ 中的 `auto` 和 `decltype` 關鍵字是什麼？
9. `new` 和 `delete` 的用法是什麼？
10. C++ 中的 `nullptr` 是什麼？

### 2. 面向對象編程

11. 面向對象編程的三大特性是什麼？
12. 解釋 C++ 中的封裝 (Encapsulation)。
13. C++ 中的繼承和多重繼承是什麼？多重繼承的優缺點？
14. 虛繼承 (Virtual Inheritance) 是什麼？為什麼需要虛繼承？
15. C++ 中的多態性 (Polymorphism) 是什麼？如何實現多態性？
16. 純虛函數 (Pure Virtual Function) 和抽象類別 (Abstract Class) 是什麼？
17. 虛函數表 (Vtable) 是什麼？它如何實現虛函數？
18. C++ 中的構造函數和析構函數的作用是什麼？
19. 複製構造函數和移動構造函數的區別是什麼？
20. 什麼是運算符重載 (Operator Overloading)？如何實現運算符重載？

### 3. C++ 模板 (Template)

21. C++ 中的模板是什麼？為什麼要使用模板？
22. 函數模板和類模板的區別是什麼？
23. 什麼是模板特化 (Template Specialization)？
24. C++ 中的模板參數類別有哪些？
25. 什麼是模板元編程 (Template Metaprogramming)？
26. 使用模板有哪些優缺點？
27. 什麼是 C++11 中的 `std::enable_if`？它的用途是什麼？
28. 什麼是 SFINAE (Substitution Failure Is Not An Error)？
29. 什麼是變參模板 (Variadic Templates)？如何使用它？
30. 如何在模板中進行偏特化 (Partial Specialization)？

### 4. 標準模板庫 (STL)

31. STL 是什麼？它包含哪些主要組件？
32. STL 中的容器有哪些？列舉一些常用的容器。
33. vector、list 和 deque 的區別是什麼？
34. 如何選擇適合的 STL 容器？
35. STL 容器中的迭代器 (Iterator) 是什麼？
36. 請解釋什麼是 STL 算法 (Algorithm)。
37. STL 中的 `std::sort` 和 `std::stable_sort` 的區別是什麼？
38. 如何使用 `std::map` 和 `std::unordered_map`？
39. C++ 中的 `std::shared_ptr` 和 `std::unique_ptr` 是什麼？
40. 什麼是智能指針 (Smart Pointer)？為什麼要使用智能指針？

### 5. 內存管理

41. C++ 中的內存分配和釋放如何進行？
42. 堆 (Heap) 和棧 (Stack) 的區別是什麼？
43. `new` 和 `malloc` 的區別是什麼？
44. 如何避免內存洩漏？常見的內存洩漏場景有哪些？
45. 什麼是 RAII (Resource Acquisition Is Initialization)？
46. C++11 中的 `move` 和 `forward` 是什麼？它們的應用場景？
47. 什麼是垃圾回收 (Garbage Collection)？C++ 支持垃圾回收嗎？
48. C++ 中的內存對齊 (Memory Alignment) 是什麼？為什麼需要內存對齊？
49. `delete` 和 `delete[]` 的區別是什麼？
50. 如何檢測和修復內存洩漏？

### 6. 異常處理

51. C++ 中的異常 (Exception) 是什麼？如何進行異常處理？
52. `try`, `catch`, `throw` 是什麼？它們的用途？
53. 如何自定義異常類別？
54. 為什麼要在 C++ 中使用異常處理？
55. C++11 中的 `noexcept` 關鍵字是什麼？
56. C++ 中異常處理的優缺點是什麼？
57. 什麼是堆棧解展 (Stack Unwinding)？
58. 為什麼不建議在構造函數中拋出異常？
59. 什麼是異常安全性 (Exception Safety)？有哪些等級？
60. 異常處理對程式效能有什麼影響？

### 7. 函數和 Lambda 表達式

61. C++ 中的函數指標和函數對象 (Function Object) 是什麼？
62. 什麼是 Lambda 表達式？如何使用？
63. Lambda 表達式中的捕獲子句 (Capture Clause) 是什麼？
64. Lambda 表達式在 C++11 和 C++14 中的改進有哪些？
65. 什麼是函數重載 (Function Overloading)？如何實現？
66. C++ 中的函數模板重載和普通函數重載有什麼區別？
67. `inline` 函數是什麼？它的優缺點是什麼？
68. C++11 中的 `std::bind` 是什麼？如何使用？
69. 如何在 C++ 中實現遞迴函數？
70. 什麼是虛函數？虛函數的作用是什麼？

### 8. 進階話題

71. C++ 中的左值 (Lvalue) 和右值 (Rvalue) 是什麼？
72. C++ 中的顯式和隱式轉換是什麼？如何防止隱式轉換？
73. C++11 中的移動語義 (Move Semantics) 是什麼？
74. 請解釋什麼是 `std::move` 和 `std::forward`。
75. C++ 中的 `static` 關鍵字有哪些應用場景？
76. `const_cast`, `static_cast`, `dynamic_cast` 和 `reinterpret_cast` 的區別是什麼？
77. C++11 中的範圍 for 循環 (Range-based for loop) 是什麼？如何使用？
78. 如何實現單例模式 (Singleton Pattern)？
79. 請解釋 C++ 中的引用折疊 (Reference Collapsing)。
80. 如何在多執行緒環境中保證資料安全？

這些問題涵蓋了 C++ 語言的各個基本知識點，可以幫助應徵者熟練掌握 C++ 的概念，為面試做好充分準備。