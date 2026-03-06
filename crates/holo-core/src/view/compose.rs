//! Composition of two `ElementWiseView` tables.
//!
//! `compose(a, b)` produces a new table where `result[i] = b.apply(a.apply(i))`.

use super::ElementWiseView;

/// Compose two views: `result[i] = other.apply(self_view.apply(i))`.
#[must_use]
pub fn compose(self_view: &ElementWiseView, other: &ElementWiseView) -> ElementWiseView {
    let a = self_view.table();
    let b = other.table();
    let mut result = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        result[i] = b[a[i] as usize];
        i += 1;
    }
    ElementWiseView::from_table(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_identity_left() {
        let id = ElementWiseView::identity();
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let composed = compose(&id, &inc);
        for i in 0..=255u8 {
            assert_eq!(composed.apply(i), inc.apply(i));
        }
    }

    #[test]
    fn compose_identity_right() {
        let id = ElementWiseView::identity();
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let composed = compose(&inc, &id);
        for i in 0..=255u8 {
            assert_eq!(composed.apply(i), inc.apply(i));
        }
    }

    #[test]
    fn compose_double_increment() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let composed = compose(&inc, &inc);
        for i in 0..=255u8 {
            assert_eq!(composed.apply(i), i.wrapping_add(2));
        }
    }

    #[test]
    fn compose_with_constant() {
        let inc = ElementWiseView::new(|x| x.wrapping_add(1));
        let c = ElementWiseView::constant(42);
        let composed = compose(&inc, &c);
        for i in 0..=255u8 {
            assert_eq!(composed.apply(i), 42);
        }
    }

    #[test]
    fn compose_associative() {
        let a = ElementWiseView::new(|x| x.wrapping_add(3));
        let b = ElementWiseView::new(|x| x.wrapping_mul(7));
        let c = ElementWiseView::new(|x| x ^ 0xAA);
        let ab_c = compose(&compose(&a, &b), &c);
        let a_bc = compose(&a, &compose(&b, &c));
        for i in 0..=255u8 {
            assert_eq!(ab_c.apply(i), a_bc.apply(i));
        }
    }
}
