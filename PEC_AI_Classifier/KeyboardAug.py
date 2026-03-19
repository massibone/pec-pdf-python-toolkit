# KeyboardAug (errori battitura, comune in PEC veloci)

keyboard_aug = nac.KeyboardAug(action="substitute", aug_char_p=0.05, aug_max=3)
augmented_keyboard = keyboard_aug.augment(pec_samples, n=3)

print("Keyboard Augmented:", augmented_keyboard)

# Output:​

# ['Fattura Eletrronica n.123...', 'Fattura Elettronica n.12;...', 'Fatutra Elettronica n.123...']
