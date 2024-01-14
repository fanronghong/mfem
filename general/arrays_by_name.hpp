// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_ARRAYS_BY_NAME
#define MFEM_ARRAYS_BY_NAME

#include "../config/config.hpp"
#include "array.hpp"

#include <iostream>
#include <map>
#include <set>
#include <string>

namespace mfem
{

template <class T>
class ArraysByName
{
protected:
   /// Map containing the data sorted alphabetically by name
   std::map<std::string,Array<T> > data;

public:

   /// Defining minimal iterators so that ranged-for loops can iterate through
   /// the data container.

   struct const_iterator;

   struct iterator
   {
      using iterator_category = std::bidirectional_iterator_tag;
      using pointer   = typename std::map<std::string,Array<T> >::iterator;
      using reference = std::pair<const std::string,Array<T> >&;

      friend struct const_iterator;

      iterator(const pointer &mit) : it(mit) {}

      reference operator*() { return *it; }
      pointer operator->() { return it; }

      // Prefix increment
      iterator& operator++() { it++; return *this; }
      // Prefix decrement
      iterator& operator--() { it--; return *this; }

      // Postfix increment
      iterator operator++(int) { iterator tmp = *this; it++; return tmp; }

      // Postfix decrement
      iterator operator--(int) { iterator tmp = *this; it--; return tmp; }

      friend bool operator== (const iterator& a, const iterator& b)
      { return a.it == b.it; };
      friend bool operator== (const iterator& a, const const_iterator& b)
      { return a.it == b.it; };
      friend bool operator!= (const iterator& a, const iterator& b)
      { return a.it != b.it; };
      friend bool operator!= (const iterator& a, const const_iterator& b)
      { return a.it != b.it; };

   private:
      typename std::map<std::string,Array<T> >::iterator it;
   };

   struct const_iterator
   {
      using iterator_category = std::bidirectional_iterator_tag;
      using pointer = typename std::map<std::string,Array<T> >::const_iterator;
      using reference = const std::pair<const std::string,Array<T> >&;

      friend struct iterator;

      const_iterator(const pointer &mit) : it(mit) {}
      const_iterator(const iterator &mit) : it(mit.it) {}

      const_iterator& operator=(const const_iterator &it2) { it = it2.it; }
      const_iterator& operator=(const iterator &it2) { it = it2.it; }

      reference operator*() const { return *it; }
      pointer operator->() { return it; }

      // Prefix increment
      const_iterator& operator++() { it++; return *this; }
      // Prefix decrement
      const_iterator& operator--() { it--; return *this; }

      // Postfix increment
      const_iterator operator++(int)
      { const_iterator tmp = *this; it++; return tmp; }

      // Postfix decrement
      const_iterator operator--(int)
      { const_iterator tmp = *this; it--; return tmp; }

      friend bool operator== (const const_iterator& a, const const_iterator& b)
      { return a.it == b.it; };
      friend bool operator== (const const_iterator& a, const iterator& b)
      { return a.it == b.it; };
      friend bool operator!= (const const_iterator& a, const const_iterator& b)
      { return a.it != b.it; };
      friend bool operator!= (const const_iterator& a, const iterator& b)
      { return a.it != b.it; };

   private:
      typename std::map<std::string,Array<T> >::const_iterator it;
   };

   /// @brief Default constructor
   inline ArraysByName() {}

   /// @brief Copy constructor: deep copy from @a src
   ///
   /// This method supports source arrays using any MemoryType.
   inline ArraysByName(const ArraysByName &src);

   /// @brief Copy constructor (deep copy) from 'src', an ArraysByName
   /// container of convertible type.
   template <typename CT>
   inline ArraysByName(const ArraysByName<CT> &src);

   /// @brief Return the number of named arrays in the container
   inline int Size() const { return data.size(); }

   /// @brief Copy the array names into an existing set of strings
   ///
   /// @note The provided set will be cleared before copying the names into it.
   inline void GetNames(std::set<std::string> &names) const;

   /// @brief Return an STL set of strings giving the names of the arrays
   inline std::set<std::string> GetNames() const;

   /// @brief Return true if an array with the given name is present in the
   /// container
   inline bool EntryExists(const std::string &name) const;

   /// @brief Reference access to the named entry.
   ///
   /// @note Passing a name for a nonexistent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline Array<T> &operator[](const std::string &name);

   /// @brief Const reference access to the named entry.
   ///
   /// @note Passing a name for a nonexistent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline const Array<T> &operator[](const std::string &name) const;

   /// @brief Create a new empty array with the given name
   ///
   /// @note Passing a name for an already existent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline Array<T> &CreateArray(const std::string &name);

   /// @brief Delete all named arrays from the container
   inline void DeleteAll();

   /// @brief Delete the named array from the container
   ///
   /// @note Passing a name for a nonexistent array will print an error
   /// message and halt execution. This is intended to call attention to
   /// possible typos or other errors. To handle such errors more gracefully
   /// consider first calling EntryExists.
   inline void DeleteArray(const std::string &name);

   /// @breif Create a copy of the internal arrays in the provided @a copy.
   inline void Copy(ArraysByName &copy) const;

   /// @brief Assignment operator: deep copy from 'src'.
   ArraysByName<T> &operator=(const ArraysByName<T> &src)
   { src.Copy(*this); return *this; }

   /// @brief Assignment operator (deep copy) from @a src, an ArraysByName
   /// container of convertible type.
   template <typename CT>
   inline ArraysByName &operator=(const ArraysByName<CT> &src);

   /// @brief Print the contents of the container to an output stream
   ///
   /// @note By default each array will be printed on its own line. A specific
   /// number of entries per line can be used by changing the @a width argument.
   void Print(std::ostream &out = mfem::out, int width = -1) const;

   /// @brief Load the contents of the container from an input stream
   ///
   /// @note This method will not first empty the container. First call
   /// DeleteAll if this behavior is needed.
   void Load(std::istream &in);

   /// @brief Sort each named array in the container
   inline void SortAll();

   /// @brief Remove duplicates from each sorted named array
   ///
   /// @note Identical entries may exist in multiple arrays but will only occur
   /// at most once in each array.
   inline void UniqueAll();

   /// STL-like begin.  Returns pointer to the first element of the array.
   inline iterator begin() { return iterator(data.begin()); }

   /// STL-like end.  Returns pointer after the last element of the array.
   inline iterator end() { return iterator(data.end()); }

   /// STL-like begin.  Returns const pointer to the first element of the array.
   inline const_iterator begin() const { return const_iterator(data.cbegin()); }

   /// STL-like end.  Returns const pointer after the last element of the array.
   inline const_iterator end() const { return const_iterator(data.cend()); }
};

template <class T>
inline bool operator==(const ArraysByName<T> &LHS, const ArraysByName<T> &RHS)
{
   if ( LHS.Size() != RHS.Size() ) { return false; }
   for (auto it1 = LHS.begin(), it2 = RHS.begin();
        it1 != LHS.end() && it2 != RHS.end(); it1++, it2++)
   {
      if (it1->first != it2->first) { return false; }
      if (it1->second != it2->second) { return false; }
   }
   return true;
}

template <class T>
inline ArraysByName<T>::ArraysByName(const ArraysByName &src)
{
   for (auto entry : src)
   {
      CreateArray(entry.first) = entry.second;
   }
}

template <typename T> template <typename CT>
inline ArraysByName<T>::ArraysByName(const ArraysByName<CT> &src)
{
   for (auto entry : src)
   {
      CreateArray(entry.first) = entry.second;
   }
}

template<class T>
inline void ArraysByName<T>::GetNames(std::set<std::string> &names) const
{
   names.clear();

   for (auto const &entry : data)
   {
      names.insert(entry.first);
   }
}

template<class T>
inline std::set<std::string> ArraysByName<T>::GetNames() const
{
   std::set<std::string> names;
   GetNames(names);
   return names;
}

template<class T>
inline bool ArraysByName<T>::EntryExists(const std::string &name) const
{
   return data.find(name) != data.end();
}

template<class T>
inline Array<T> &ArraysByName<T>::operator[](const std::string &name)
{
   MFEM_VERIFY( data.find(name) != data.end(),
                "Access to unknown named array \"" << name << "\"");
   return data[name];
}

template<class T>
inline const Array<T> &ArraysByName<T>::operator[](const std::string &name)
const
{
   MFEM_VERIFY( data.find(name) != data.end(),
                "Access to unknown named array \"" << name << "\"");
   return data.at(name);
}

template<class T>
inline Array<T> &ArraysByName<T>::CreateArray(const std::string &name)
{
   MFEM_VERIFY( data.find(name) == data.end(),
                "Named array \"" << name << "\" already exists");
   Array<T> empty_array;
   data.insert(std::pair<std::string,Array<T> >(name,empty_array));
   return data[name];
}

template<class T>
inline void ArraysByName<T>::DeleteAll()
{
   data.clear();
}

template<class T>
inline void ArraysByName<T>::DeleteArray(const std::string &name)
{
   MFEM_VERIFY( data.find(name) != data.end(),
                "Attempting to delete unknown named array \"" << name << "\"");
   data.erase(name);
}

template<class T>
inline void ArraysByName<T>::Copy(ArraysByName &copy) const
{
   copy.DeleteAll();
   for (auto entry : data)
   {
      copy.CreateArray(entry.first) = entry.second;
   }
}

template <typename T> template <typename CT>
inline ArraysByName<T> &ArraysByName<T>::operator=(const ArraysByName<CT> &src)
{
   DeleteAll();
   for (auto entry : src)
   {
      CreateArray(entry.first) = entry.second;
   }
   return *this;
}

template <class T>
inline void ArraysByName<T>::SortAll()
{
   for (auto a : data)
   {
      a.second.Sort();
   }
}

template <class T>
inline void ArraysByName<T>::UniqueAll()
{
   for (auto a : data)
   {
      a.second.Unique();
   }
}

}

#endif

